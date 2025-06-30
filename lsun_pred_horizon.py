import pdb
import sys
import cv2
import os
import json
import argparse
import numpy as np

from tqdm import tqdm
from imageio import imwrite
from utils.pp_utils import read_image, read_json, tell_type, normalize_corners, gen_1d
from external.PerspectiveFields.perspective2d import PerspectiveFields
from external.PerspectiveFields.perspective2d.utils import general_vfov_to_focal
from external.PerspectiveFields.perspective2d.utils.panocam import PanoCam

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aligned_data_dir',
        type=str,
        default='src/pp/lsun_align',
        help='Path to the data directory'
    )

    parser.add_argument(
        '--original_data_dir',
        type=str,
        default='src/pp/lsun_ori/images_ori',
        help='Path to the original data directory'
    )

    parser.add_argument(
        '--data_mode',
        type=str,
        default='val',
        help='train or val'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='src/pp/lsun_align',
        help='Path to the output directory'
    )

    args = parser.parse_args()

    return args

def main(args):
    PerspectiveFields.versions()

    version = 'Paramnet-360Cities-edina-centered'
    # version = 'Paramnet-360Cities-edina-uncentered'
    # version = 'PersNet_Paramnet-GSV-centered'
    # version = 'PersNet_Paramnet-GSV-uncentered'
    # version = 'PersNet-360Cities'
    pf_model = PerspectiveFields(version).eval().cuda()

    aligned_data_dir = args.aligned_data_dir
    data_mode = args.data_mode
    output_path = args.output_path
    output_path = os.path.join(output_path, data_mode)
    os.makedirs(output_path, exist_ok=True)
    img_dir = os.path.join(aligned_data_dir, data_mode, 'img_aligned')
    gt_dir = os.path.join(aligned_data_dir, data_mode, 'cor_aligned')
    pano_gt_dir = os.path.join(aligned_data_dir, data_mode, 'boundary_aligned_pano')
    scene_list_dir = os.path.join(aligned_data_dir, data_mode, f'{data_mode}_scene_list.json')
    with open(scene_list_dir, 'r') as f:
        data_lst = json.load(f)
    #  for image that contain ceiling and floor, we utilize ground truth to calculate horizon
    #  for image that only contain ceiling or floor, we utilize PerspectiveFields to calculate horizon
    data_list = data_lst['only_floor'] + data_lst['only_ceiling']

    pp_shape = (512,512)
    data_horizon = {}
    
    for data in tqdm(data_list, desc="compute predict horizon"):
        img_path = os.path.join(img_dir, f"{data}.png")
        gt_path = os.path.join(gt_dir, f"{data}.json")
        pano_gt_path = os.path.join(pano_gt_dir, f"{data}.json")
        img = read_image(img_path, pp_shape)
        gt = read_json(gt_path)
        pano_gt = read_json(pano_gt_path)
        scale =  np.array(gt["new"]["hw"]).reshape(-1, 2).astype(np.float32)[0]
        o_scale =  np.array(gt["ori"]["hw"]).reshape(-1, 2).astype(np.float32)[0]
        ceiling_corner = np.array(gt["new"]["ceiling_corner"]).reshape(-1, 2).astype(np.float32)
        floor_corner = np.array(gt["new"]["floor_corner"]).reshape(-1, 2).astype(np.float32)
        o_ceiling_corner = np.array(gt["ori"]["ceiling_corner"]).reshape(-1, 2).astype(np.float32)
        o_floor_corner = np.array(gt["ori"]["floor_corner"]).reshape(-1, 2).astype(np.float32)
        gt_type = tell_type(ceiling_corner, floor_corner)
        o_gt_type = tell_type(o_ceiling_corner, o_floor_corner)
        scale, ceiling_corner, floor_corner = normalize_corners(scale, gt_type, ceiling_corner, floor_corner)
        o_scale, o_ceiling_corner, o_floor_corner = normalize_corners(o_scale, o_gt_type, o_ceiling_corner, o_floor_corner)

        c_1d = gen_1d(ceiling_corner, pp_shape, missing_val=-1, mode='constant')[None,...]
        f_1d = gen_1d(floor_corner, pp_shape, missing_val=(img.shape[0]*1.01), mode='constant')[None,...]
        ori_c_1d = gen_1d(o_ceiling_corner, pp_shape, missing_val=-1, mode='constant')[None,...]
        ori_f_1d = gen_1d(o_floor_corner, pp_shape, missing_val=(img.shape[0]*1.01), mode='constant')[None,...]
        if gt_type == 0:
            horizon_aligned = (np.max(c_1d) + np.min(f_1d)) / 2
        if o_gt_type == 0:
            ori_horizon = (np.max(ori_c_1d) + np.min(ori_f_1d)) / 2

        ########## pano gt ##########
        pano_boundary = np.array(pano_gt['boundary'])
        pano_boundary = ((pano_boundary / np.pi + 0.5) * 512).astype(np.int16)
        max_c_b = np.max(pano_boundary[0])
        min_f_b = np.min(pano_boundary[1])
        ##############################

        original_img_path = os.path.join(args.original_data_dir, f"{data}.jpg")
        o_img = cv2.imread(original_img_path)
        predictions = pf_model.inference(img_bgr=o_img)
        #### calculate horizon pixel error ####
        roll = predictions["pred_roll"].item()
        pitch = predictions["pred_pitch"].item()
        vfov = predictions["pred_general_vfov"].item()
        rel_cx = predictions["pred_rel_cx"].item()
        rel_cy = predictions["pred_rel_cy"].item()
        # degree to radian
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        vfov = np.radians(vfov)
        # calculate latitudes
        rel_focal = general_vfov_to_focal(rel_cx, rel_cy, 1, vfov, False)
        lati_deg = PanoCam.get_lat_general(
            focal_rel=rel_focal,
            im_w=o_img.shape[1],
            im_h=o_img.shape[0],
            elevation=pitch,
            roll=roll,
            cx_rel=rel_cx,
            cy_rel=rel_cy,
        )
        horizon_o_pred = (np.where((lati_deg < 1) & (lati_deg > -1))[0]).mean()
        pitch = np.rad2deg(pitch)
        if np.isnan(horizon_o_pred):
            if pitch > 0:
                horizon_o_pred = pp_shape[0]
            elif pitch < 0:
                horizon_o_pred = 0
            else:
                horizon_o_pred = pp_shape[0] / 2
        else:
            horizon_o_pred = horizon_o_pred / o_img.shape[0] * pp_shape[0]
        data_horizon[data] = [horizon_o_pred]

    with open(os.path.join(output_path, f'{data_mode}_lsun_pred_horizon.json'), 'w') as f:
        json.dump(data_horizon, f, indent=4)
        

    
if __name__ == '__main__':
    args = get_args()
    main(args)