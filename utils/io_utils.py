import csv
import json
import logging
import os
import cv2
import shutil
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from utils.EquirecCoordinate import XY2xyz, xyz2XY, xyz2lonlat, lonlat2xyz, EquirecTransformer


def get_abs_path(file):
    return os.path.dirname(os.path.abspath(file))

def create_directory(output_dir, delete_prev=True, ignore_request=False):
    if os.path.exists(output_dir) and delete_prev:
        if not ignore_request:
            logging.warning(f"This directory will be deleted: {output_dir}")
            input("This directory will be deleted. PRESS ANY KEY TO CONTINUE...")
        shutil.rmtree(output_dir, ignore_errors=True)
    if not os.path.exists(output_dir):
        logging.info(f"Dir created: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    return Path(output_dir).resolve()


def save_json_dict(filename, dict_data):
    with open(filename, "w") as outfile:
        json.dump(dict_data, outfile, indent="\t")


def print_cfg_information(cfg):
    logging.info(f"Experiment ID: {cfg.id_exp}")
    logging.info(f"Output_dir: {cfg.output_dir}")

def save_cfg(cfg_file, cfg):
    with open(cfg_file, "w") as fn:
        OmegaConf.save(config=cfg, f=fn)

def plotXY(rgb, XY, color, thickness=4):
    for i in range(XY.shape[0]-1):
        a = XY[i, ...].round().astype(int)
        b = XY[i+1, ...].round().astype(int)
        if abs(a[0] - b[0]) > 0.5 * rgb.shape[1] or a[1] < 0 or a[1] > rgb.shape[0] or b[1] < 0 or b[1] > rgb.shape[0]:
            continue
        else:
            #pdb.set_trace()
            cv2.line(rgb, tuple(a), tuple(b), color=color, thickness=thickness)

def interpolate(a, b, count=100):
    x = np.linspace(a[0], b[0], count)[:, None]
    y = np.linspace(a[1], b[1], count)[:, None]
    z = np.linspace(a[2], b[2], count)[:, None]
    xyz = np.concatenate([x, y, z], axis=-1)

    return xyz

def prepare_corner_wo_occlusion(corner, ratio=1, ch=1.6, shape=[512,1024]):
    scale = 300
    C_XY = np.array([[0, 0]])
    F_XY = np.array([[0, 0]])
    corner_num = len(corner) // 2
    ET = EquirecTransformer('numpy')
    cor_xyz = ET.XY2xyz(corner, shape=[512,1024])
    floor_pts = cor_xyz[1::2].T
    ratio_f = ch / floor_pts[1]
    floor_pts = floor_pts * ratio_f

    ceiling_pts = floor_pts.copy()
    ceiling_pts[1] = -(ratio * ch)

    floor_pts = floor_pts.T
    ceiling_pts = ceiling_pts.T
    
    for i in range(corner_num):

        cp1 = ceiling_pts[i].copy()

        fp1 = floor_pts[i].copy()
        #fp1[1] = label['cameraHeight']
        #fp1[1] = 1.6
        if i==(corner_num-1):
            cp2 = ceiling_pts[0].copy()
            fp2 = floor_pts[0].copy()
        else:
            cp2 = ceiling_pts[i+1].copy()
            fp2 = floor_pts[i+1].copy()
        #fp2[1] = label['cameraHeight']
        #fp2[1] = 1.6
        norm_cp12 = np.linalg.norm(cp1-cp2)
        pts_num = np.around(norm_cp12*scale).astype(int)
        if i==0:
            ceiling_xyz = interpolate(cp1, cp2, count=pts_num)
            floor_xyz = interpolate(fp1, fp2, count=pts_num)

        else:
            ceiling_xyz = np.append(ceiling_xyz, interpolate(cp1, cp2, count=pts_num), axis=0)
            floor_xyz = np.append(floor_xyz, interpolate(fp1, fp2, count=pts_num), axis=0)

    ceiling_XY = ET.xyz2XY(ceiling_xyz, shape=shape).round().astype(int)
    floor_XY = ET.xyz2XY(floor_xyz, shape=shape).round().astype(int)

    return ceiling_XY, floor_XY

