import pdb
import sys
import os
import json
import argparse
import hydra
import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf
from imageio import imwrite
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
from utils.pp_utils import read_image, gen_1d, read_json, tell_type, \
                           whole_process_pp2pano, normalize_corners


@hydra.main(version_base="1.3", config_path="config/.", config_name="config")
def main(cfg: DictConfig):
    cfg.lsun.data_dir = os.path.join(root_path, cfg.lsun.data_dir)
    output_new_data_list_dir = os.path.join(cfg.out_dir, cfg.lsun.mode)
    output_img_dir = os.path.join(output_new_data_list_dir, "img_aligned_pano")
    output_boundary_dir = os.path.join(output_new_data_list_dir, "boundary_aligned_pano")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_boundary_dir, exist_ok=True)
    img_dir = cfg.lsun.img_dir
    gt_dir = cfg.lsun.gt_dir
    scene_list_dir = cfg.lsun.scene_list_dir
    # param setting
    pp_shape = cfg.param.pp_shape
    pano_shape = cfg.param.pano_shape
    out_bound_val = cfg.param.out_bound_val
    pano_rotate_angle = cfg.param.pano_rotate_angle

    with open(scene_list_dir, 'r') as f:
        data_lst = json.load(f)
    data_list = data_lst['ceiling_floor'] + data_lst['only_ceiling'] + data_lst['only_floor']
    new_data_list = {'ceiling_floor':[], 'only_ceiling':[], 'only_floor':[], 'error_data':[]}
    # process data
    for data in tqdm(data_list):
        img_path = os.path.join(img_dir, f"{data}.png")
        gt_path = os.path.join(gt_dir, f"{data}.json")
        img = read_image(img_path, pp_shape)
        gt = read_json(gt_path)
        scale =  np.array(gt["new"]["hw"]).reshape(-1, 2).astype(np.float32)[0]
        ceiling_corner = np.array(gt["new"]["ceiling_corner"]).reshape(-1, 2).astype(np.float32)
        floor_corner = np.array(gt["new"]["floor_corner"]).reshape(-1, 2).astype(np.float32)
        gt_type = tell_type(ceiling_corner, floor_corner)
        scale, ceiling_corner, floor_corner = normalize_corners(scale, gt_type, ceiling_corner, floor_corner)
        try:
            c_1d = gen_1d(ceiling_corner, pp_shape, missing_val=-1, mode='constant')[None,...]
            f_1d = gen_1d(floor_corner, pp_shape, missing_val=(img.shape[0]*1.01), mode='constant')[None,...]
        except:
            print(f"Error: {data}")
            new_data_list['error_data'].append(data)
            continue
        boundary_pp = np.concatenate((c_1d, f_1d), axis=0).astype(np.int16)

        equi_img, boundary = whole_process_pp2pano(img, boundary_pp, pp_shape, gt_type=gt_type, \
                                                    pano_rotate_angle=pano_rotate_angle, \
                                                    out_bound_val=out_bound_val)
        # save img_name in new_data_list
        if gt_type == 0:
            new_data_list['ceiling_floor'].append(data)
        elif gt_type == 1:
            new_data_list['only_ceiling'].append(data)
        elif gt_type == 2:
            new_data_list['only_floor'].append(data)
        else:
            ValueError("gt_type is wrong, please check it.")
        # save img
        imwrite(os.path.join(output_img_dir, f"{data}.png"), (equi_img*255).permute(1,2,0).numpy().astype(np.uint8))
        # save boundary json
        output_gt = dict()
        output_gt["name"] = data
        output_gt["gt_type"] = gt_type
        output_gt["boundary"] = boundary.tolist()
        with open(os.path.join(output_boundary_dir, f"{data}.json"), 'w') as f:
            json.dump(output_gt, f, indent=4)

    # save new_data_list
    with open(os.path.join(output_new_data_list_dir, f"{cfg.lsun.mode}_new_data_list.json"), 'w') as f:
        json.dump(new_data_list, f, indent=4)

    

if __name__ == "__main__":
    main()