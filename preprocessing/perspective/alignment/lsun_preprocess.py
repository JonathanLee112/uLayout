import pdb
import hydra
import os
import argparse
import numpy as np

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
# LSUN_keypoint_struc represents the different type of the LSUN ground truth
# we utilize the image and ground truth after resizing and separating the corner into ceiling, floor, ceiling-wall, and floor-wall
from utils.utility import load_lsun_mat, read_image, resize_keypoints, LSUN_keypoint_struc
from utils.preproc_utils import extract_edgelets, prepare_vp, extract_z_vp, align_by_vpz

'''
This perspective preprocessing process is following by the paper Flat2Layout https://arxiv.org/abs/1905.12571
'''
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

@hydra.main(version_base="1.3", config_path="config/.", config_name="config")
def main(cfg: DictConfig):
    output_dir = cfg.output_dir
    # lsun resized data
    img_dir = os.path.join(root_path, cfg.lsun.img_dir)
    data_mode = cfg.lsun.data_mode
    label_path = os.path.join(root_path, cfg.lsun.label_path)
    ori_img_shape = cfg.lsun.img_shape
    gt = np.load(label_path, allow_pickle=True)
    img_list = [str(i) for i in gt['img_name']]
    corner_c = [np.array(c).reshape(-1, 2) for c in gt['ceiling_corner']]
    corner_f = [np.array(c).reshape(-1, 2) for c in gt['floor_corner']]
    corner_cw = [np.array(c).reshape(-1, 2) for c in gt['ceiling_wall']]
    corner_fw = [np.array(c).reshape(-1, 2) for c in gt['floor_wall']]
    
    save = False
    visual = False
    if cfg.save_pd:
        save = True
    if cfg.vis:
        visual = True
    # parameters
    img_shape = cfg.param.shape
    quant = cfg.param.quant
    skip_pad = cfg.param.skip_pad
    vp_thres = cfg.param.vp_thres
    deg_thres = cfg.param.deg_thres
    ransac_it = cfg.param.ransac_it

    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(len(img_list)), desc='Processing LSUN data'):
        img_id = img_list[idx]
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        img = read_image(img_path, img_shape)
        c_c = corner_c[idx]
        c_f = corner_f[idx]
        c_cw = corner_cw[idx]
        c_fw = corner_fw[idx]
        c_c, c_f, c_cw, c_fw = resize_keypoints(c_c, c_f, c_cw, c_fw, ori_img_shape, img_shape)
        edgelets = extract_edgelets(img_id, img, output_dir, quant, skip_pad, save)
        #z_res, x_res, y_res = prepare_vp(img_id, img, c_c, c_f, c_cw, c_fw, output_dir, save, visual)
        vpz = extract_z_vp(img_id, img, c_cw, c_fw, output_dir, edgelets,
                           vp_thres, deg_thres, ransac_it, save, visual)
        align_by_vpz(img_id, img, c_c, c_f, c_cw, c_fw, output_dir, data_mode, vpz, visual)
        



if __name__ == '__main__':
    main()