import sys
import os
import json
import numpy as np
import cv2
import torch
import pathlib
import pdb

from scipy.io import loadmat
from imageio import imread
from collections import namedtuple
from pathlib import Path
from typing import Sequence
from utils.Cube2Equirec import Cube2Equirec

# from datasets.panostretch import pano_stretch
# from datasets.mp3d_hn_ori_dataset import cor2xybound
from utils.eval_utils_research import phi_coords2xyz, phi_coords2xyz_torch

Scene = namedtuple('Scene', ['filename', 'scene_type', 'layout_type', 'keypoints', 'shape'])

def extract_seg_index(seg, gt_type):
    h, _ = seg.shape
    seg_index = np.zeros((2,5))
    for i in range(5):
        index = np.where(seg == i+1)
        if len(index[0]) > 0:
            seg_index[0,i] = np.mean(index[0])
            seg_index[1,i] = max(index[0])
        else:
            seg_index[0,i] = h / 2
    if gt_type == 1:
        c_idx = np.argmin(seg_index[0]) + 1
        f_idx = 0
    elif gt_type == 2:
        c_idx = 0
        f_idx = np.argmax(seg_index[0]) + 1
    else:
        c_idx = np.argmin(seg_index[0]) + 1
        if seg_index[1, c_idx-1] > h / 2:
            c_idx = 0
        f_idx = np.argmax(seg_index[0]) + 1

    return c_idx, f_idx


def seg2boundary(seg, c_idx=5, f_idx=4, normalize=True):
    '''
    lsun segmentation label
    1:Frontal wall  2:Left wall  3:Right wall  4:Floor  5:Ceiling
    mp3d-layout segmentation label
    1:ceiling  3:floor

    similar to Flat2Layout Hsiao et al. (2019)
    boundary is normalized in range [-0.5, 0.5]
    ceiling boundary which is out of image is set to -0.51
    floor boundary which is out of image is set to 0.51 
    '''
    h, w = seg.shape
    boundary = np.zeros((2,w))

    for i in range(w):
        #extract ceiling boundary
        c_index = np.where(seg[:,i] == c_idx)[0]
        if len(c_index) == 0:
            if normalize:
                boundary[0,i] = -0.51
            else:
                boundary[0,i] = -(h-1)*0.01
        else:
            if normalize:
                boundary[0,i] = c_index[-1] / (h-1) - 1/2
            else:
                boundary[0,i] = c_index[-1]
        
        #extract floor boundary
        f_index = np.where(seg[:,i] == f_idx)[0]
        if len(f_index) == 0:
            if normalize:
                boundary[1,i] = 0.51
            else:
                boundary[1,i] = (h-1)*1.01
        else:
            if normalize:
                boundary[1,i] = f_index[0] / (h-1) - 1/2
            else:
                boundary[1,i] = f_index[0]

    return boundary.astype(np.float32)

def read_image(img_path, shape=None):
    img = imread(img_path, pilmode='RGB').astype(np.float32) / 255
    if shape is not None:
        if img.shape[0] != shape[0] or img.shape[1] != shape[1]:
            img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_AREA)
    
    return img

def read_seg(seg_path,shape=None):
    if seg_path.endswith('.mat'):
        seg = loadmat(seg_path)['layout']
    elif seg_path.endswith('.png') or seg_path.endswith('.jpg'):
        seg = imread(seg_path, pilmode='RGB').astype(np.int16)
    else:
        raise ValueError('Unknown segmentation format')

    if shape is not None:
        if seg.shape[0] != shape[0] or seg.shape[1] != shape[1]:
            seg = cv2.resize(seg, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    
    return seg.astype(np.int16)

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

def tell_type(c_c, f_c):
    '''
    input:
        c_c: ceiling corner
        f_c: floor corner
    output:
        gt_type : 0 for both, 1 for ceiling, 2 for floor
    '''
    if len(c_c) == 0 and len(f_c) == 0:
        return ValueError('No ceiling and floor corner!')
    elif len(c_c) == 0:
        gt_type = 2
    elif len(f_c) == 0:
        gt_type = 1
    else:
        gt_type = 0

    return gt_type

def tell_type_pp_model(gt_boundary, out_bound_val=75):
    '''
    input:
        gt_boundary: (2, w)
    output:
        gt_type : 0 for both, 1 for ceiling, 2 for floor
    '''
    out_bound_val = np.deg2rad(out_bound_val)
    out_c_idx = np.where(gt_boundary[0] == (-1*out_bound_val))[0]
    out_f_idx = np.where(gt_boundary[1] == out_bound_val)[0]
    w = gt_boundary.shape[1]

    if len(out_c_idx) == w and len(out_f_idx) == w:
        return ValueError('No ceiling and floor boundary!')
    elif len(out_c_idx) == w:
        gt_type = 2
    elif len(out_f_idx) == w:
        gt_type = 1
    else:
        gt_type = 0

    return gt_type


def normalize_corners(scale, gt_type, c_c, f_c):
    if gt_type == 0:
        if c_c[0,0] < 0:
            c_c[0,0] = 0
        if f_c[0,0] < 0:
            f_c[0,0] = 0
        if max(c_c[-1,0], f_c[-1,0]) > scale[1]:
            scale[1] = max(c_c[-1,0], f_c[-1,0])
        c_c[:,0] /= scale[1]
        c_c[:,1] /= scale[0]
        f_c[:,0] /= scale[1]
        f_c[:,1] /= scale[0]
    elif gt_type == 1:
        if c_c[0,0] < 0:
            c_c[0,0] = 0
        if c_c[-1,0] > scale[1]:
            scale[1] = c_c[-1,0]
        c_c[:,0] /= scale[1]
        c_c[:,1] /= scale[0]
    elif gt_type == 2:
        if f_c[0,0] < 0:
            f_c[0,0] = 0
        if f_c[-1,0] > scale[1]:
            scale[1] = f_c[-1,0]
        f_c[:,0] /= scale[1]
        f_c[:,1] /= scale[0]

    return scale, c_c, f_c

def load_lsun_mat(filepath: pathlib.Path) -> Sequence[Scene]:
    data = loadmat(filepath)

    return [
        Scene(*(m.squeeze() for m in metadata))
        for metadata in data[Path(filepath).stem].squeeze()
    ]

# def panostretch(img, c_map, f_map, max_stretch=2.0):
#     img = img.permute(1, 2, 0).cpu().numpy()
#     c_map = c_map[...,None].cpu().numpy()
#     f_map = f_map[...,None].cpu().numpy()
#     uv_index = np.where(img > np.array([0,0,0]))
#     v_min = np.min(uv_index[1])
#     v_max = np.max(uv_index[1])
#     x_range = np.linspace(v_min,v_max, (v_max-v_min+1)).astype(np.int16)
#     boundary = np.zeros((2, x_range.shape[0]))
#     for i in x_range:
#         index = np.where(img[:,i,:] > np.array([0,0,0]))
#         if len(index[0]) > 0:
#             boundary[0,i - x_range[0]] = index[0][0]
#             boundary[1,i - x_range[0]] = index[0][-1]
#         else:
#             continue
#     cor_y_boun = [[boundary[0,0],boundary[0,-1]]]
#     x = [[x_range[0],x_range[-1]]]
#     cor = np.vstack((x, cor_y_boun))
#     #cor = np.vstack((x_range, boundary[0]))
#     xmin, xmax, ymin, ymax = cor2xybound(cor)
#     # kx = np.random.uniform(1.0, max_stretch)
#     # ky = np.random.uniform(1.0, max_stretch)
#     kx = 2
#     ky = 1
#     kx = max(1 / kx, min(0.5 / xmin, 1.0))
#     ky = max(1 / ky, min(0.5 / ymin, 1.0))
#     # if np.random.randint(2) == 0:
#     #     kx = max(1 / kx, min(0.5 / xmin, 1.0))
#     # else:
#     #     kx = min(kx, max(10.0 / xmax, 1.0))
#     # if np.random.randint(2) == 0:
#     #     ky = max(1 / ky, min(0.5 / ymin, 1.0))
#     # else:
#     #     ky = min(ky, max(10.0 / ymax, 1.0))
#     img, _ = pano_stretch(img, cor, kx, ky)
#     c_map, _ = pano_stretch(c_map, cor, kx, ky)
#     f_map, _ = pano_stretch(f_map, cor, kx, ky)

#     return img, c_map, f_map

def aug_flip(*corners_lst):
        for cs in corners_lst:
            if len(cs) == 0:
                continue
            cs[:, 0] = 1 - cs[:, 0]  # Flip x
            cs[:] = np.flip(cs, 0)
        return [*corners_lst]

def aug_h_roll(equi_img, boundary, pp_shape=(256,256), pano_shape=(512,1024)):
    '''
    Input:
        equi_img: equirectangular image (3, h, w)
        boundary: (2, w)
        pp_shape: (h, w)
        pano_shape: (h, w)
    Output:
        equi_img: random horizontal rolled equirectangular image (3, h, w)
        boundary: random horizontal rolled rotated boundary (2, w)
    '''
    bound = pano_shape[1] / 2 - pp_shape[1] / 2
    shift = np.random.randint(-bound, bound)
    color_u_idx = np.where(equi_img.sum(axis=(0,1)) != 0)[0]
    #color_u_idx = torch.nonzero(equi_img.sum(dim=(0,1)) != 0)
    mid = min(color_u_idx) + max(color_u_idx) // 2
    if (mid + pano_shape[1] / 2 + shift) > (pano_shape[1] - 1):
        # 128 is half of pp_shape[1]
        shift = int(pano_shape[1] - 1 - mid - pp_shape[1] / 2)
    elif (mid - pano_shape[1] / 2 + shift) < 0:
        shift = int(-1 * mid + pp_shape[1] / 2)
    equi_img = np.roll(equi_img, shift, axis=2)
    boundary = np.roll(boundary, shift, axis=1)

    return equi_img, boundary

def gen_1d(xys, shape, missing_val, mode='constant'):
    '''  generate 1d boundary GT
    Input:
        xys: xy coordinates of keypoints

    Mode: setting value of y when it is outside image plane
        constant: set to a constant, missing_val
        linear: linearly grow to missing_val
    '''
    h, w = shape
    reg_1d = np.zeros(w, np.float32) + missing_val
    for i in range(1, len(xys)):
        x0, y0 = xys[i-1]
        x1, y1 = xys[i]
        x0 = x0 * w           # [0, 1] => [0, w]
        x1 = x1 * w           # [0, 1] => [0, w]
        # y0 = (y0 - 0.5) * 2   # [0, 1] => [-1, 1]
        # y1 = (y1 - 0.5) * 2   # [0, 1] => [-1, 1]
        y0 = y0 * h           # [0, 1] => [0, h]
        y1 = y1 * h           # [0, 1] => [0, h]
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        s = int(max(x0-1, 0))
        e = int(min(x1+1, w))
        reg_1d[s:e] = np.interp(np.arange(s, e), [x0, x1], [y0, y1])

    if len(xys)>0 and mode == 'linear':
        xst = 0
        x0, y0 = xys[0]
        if x0 >= 0:
            x1, y1 = xys[1]
            x0 = x0 * w           # [0, 1] => [0, w]
            x1 = x1 * w           # [0, 1] => [0, w]
            y0 = (y0 - 0.5) * 2   # [0, 1] => [-1, 1]
            y1 = (y1 - 0.5) * 2   # [0, 1] => [-1, 1]
            if x0-x1==0:
                raise ValueError('x0-x1==0')
            yst = y0+(y0-y1)/(x0-x1)*(xst-x0)
            #  if len(np.arange(xst, int(x0))) != len(reg_1d[xst:int(x0)]):
                #  print('corners:', xys)
                #  print('xst, yst:', xst, yst)
                #  print('x0, y0:', x0, y0)
            reg_1d[xst:int(x0)] = np.interp(np.arange(xst, int(x0)), [xst, x0], [yst, y0])

        xend = int(w)
        x0, y0 = xys[-2]
        x1, y1 = xys[-1]
        if x1 <= w:
            x0 = x0 * w           # [0, 1] => [0, w]
            x1 = x1 * w           # [0, 1] => [0, w]
            y0 = (y0 - 0.5) * 2   # [0, 1] => [-1, 1]
            y1 = (y1 - 0.5) * 2   # [0, 1] => [-1, 1]
            yend = y0+(y0-y1)/(x0-x1)*(xend-x0)
            reg_1d[int(min(x1+1, w)):xend] = np.interp(np.arange(int(min(x1+1, w)), xend), [x1, xend], [y1, yend])
        if missing_val > 1:
            reg_1d = np.clip(reg_1d, None, missing_val)
        else:
            reg_1d = np.clip(reg_1d, missing_val, None)

    return reg_1d

def pp_boundary2map(pp_boundary, img_shape=(256,256)):
    h, w = img_shape
    b_map_c = np.zeros((3, h, w))
    b_map_f = np.zeros((3, h, w))

    valid_c = np.logical_and(pp_boundary[0] >= 0, pp_boundary[0] < h)
    valid_f = np.logical_and(pp_boundary[1] >= 0, pp_boundary[1] < h)

    b_map_c[:, pp_boundary[0, valid_c], np.arange(w)[valid_c]] = 1
    b_map_f[:, pp_boundary[1, valid_f], np.arange(w)[valid_f]] = 1

    return b_map_c, b_map_f

def pp2pano(img, b_map_c, b_map_f, pano_shape=(512,1024), pp_shape=(256, 256), gt_type=0):
    '''
    Input:
        img: RGB image (256,256,3)
        b_map_c: ceiling boundary map
        b_map_f: floor boundary map
        pano_shape: (h, w)
        pp_shape: (h, w)
        gt_type: 0 for both, 1 for ceiling, 2 for floor

    Output:
        equi_img: equirectangular image
        equi_b_c: ceiling boundary map
        equi_b_f: floor boundary map
    '''
    c2e = Cube2Equirec(pano_shape[0], pano_shape[1], pp_shape[1], CUDA=True)
    img = torch.FloatTensor(img.copy()).permute(2, 0, 1)[None, ...]
    zero_img = torch.zeros_like(img)
    image_tensor = torch.cat((zero_img, zero_img, img, zero_img, zero_img, zero_img), dim=0)
    equi_img = c2e(image_tensor)[0]

    if gt_type == 1:
        b_c_tensor = torch.zeros_like(image_tensor)
        b_c_tensor[2] = torch.FloatTensor(b_map_c)
        equi_b_c = c2e(b_c_tensor)[0,0]
        equi_b_f = torch.zeros_like(equi_b_c)
    elif gt_type == 2:
        b_f_tensor = torch.zeros_like(image_tensor)
        b_f_tensor[2] = torch.FloatTensor(b_map_f)
        equi_b_f = c2e(b_f_tensor)[0,0]
        equi_b_c = torch.zeros_like(equi_b_f)
    elif gt_type == 0:
        b_c_tensor = torch.zeros_like(image_tensor)
        b_c_tensor[2] = torch.FloatTensor(b_map_c)
        equi_b_c = c2e(b_c_tensor)[0,0]
        b_f_tensor = torch.zeros_like(image_tensor)
        b_f_tensor[2] = torch.FloatTensor(b_map_f)
        equi_b_f = c2e(b_f_tensor)[0,0]
    else:
        raise ValueError('gt_type should be ceiling, floor or both, 0 for both, 1 for ceiling, 2 for floor')

    
    return equi_img, equi_b_c, equi_b_f

def pano_boundary2map(equi_b_c, equi_b_f, pano_shape=(512,1024), out_bound_val=75):
    '''
    Input:
        equi_b_c: ceiling boundary map (h, w)
        equi_b_f: floor boundary map (h, w)
        pano_shape: (h, w)
        out_bound_val: value of boundary outside image plane

    Output:
        boundary: (2, w)
    '''

    out_bound_val = np.deg2rad(out_bound_val)
    h, w = pano_shape
    boundary = np.zeros((2,w))

    c_index = np.where(equi_b_c > 0)
    f_index = np.where(equi_b_f > 0)

    c_boundary = np.argmax(equi_b_c, axis=0)
    f_boundary = np.argmax(equi_b_f, axis=0)

    boundary[0, c_index[1]] = (c_boundary[c_index[1]] / h - 0.5) * np.pi
    boundary[1, f_index[1]] = (f_boundary[f_index[1]] / h - 0.5) * np.pi

    no_c_index = np.setdiff1d(np.arange(w), c_index[1])
    no_f_index = np.setdiff1d(np.arange(w), f_index[1])

    boundary[0, no_c_index] = -1 * out_bound_val
    boundary[1, no_f_index] = out_bound_val


    return boundary

def whole_process_pp2pano(pp_img, pp_boundary, pp_shape=(256,256), pano_shape=(512,1024), gt_type=0, pano_rotate_angle=30, out_bound_val=75):
    '''
    Input:
        pp_img: RGB image (256,256,3)
        pp_boundary: (2, w)
        pp_shape: (h, w)
        pano_shape: (h, w)
        gt_type: 0 for both, 1 for ceiling, 2 for floor
        pano_rotate_angle: rotation angle
        out_bound_val: value of boundary outside image plane
    output:
        equi_img: equirectangular image
        boundary: (2, w)
    '''
    assert gt_type in [0, 1, 2], 'gt_type should be [0, 1, 2], 0 for both, 1 for ceiling, 2 for floor'
    b_map_c, b_map_f = pp_boundary2map(pp_boundary, pp_shape)
    equi_img, equi_b_c, equi_b_f = pp2pano(pp_img, b_map_c, b_map_f, pano_shape, pp_shape, gt_type)
    boundary = pano_boundary2map(equi_b_c, equi_b_f, pano_shape, out_bound_val)

    return equi_img, boundary

def rolling_img_boundary(equi_img, boundary, eval_range, gt_type, thres=60, pano_shape=(512,1024), pp_shape=(256,256)):
    '''
    Input:
        equi_img: equirectangular image (3, h, w)
        boundary: (2, w)
        eval_range: (2, w)
        gt_type: 0 for both, 1 for ceiling, 2 for floor
        thres: threshold of boundary
        pano_shape: (h, w)
    output:
        equi_img: rolled equirectangular image
        boundary: rolled boundary
        diff_pixel: number of pixels rolled
    '''
    h, _ = pano_shape
    h_pp, _ = pp_shape
    thres = np.deg2rad(thres)
    diff_lonlat = 0
    diff_pixel = 0
    eval_c_idx = np.where(eval_range[0] != 0)[0]
    eval_f_idx = np.where(eval_range[1] != 0)[0]
    if gt_type == 1:
        c_idx = np.where((boundary[0] == thres) | (boundary[0] > -thres))[0]
        if len(c_idx) > 0:
            diff_lonlat = max(boundary[0]) - (-thres)
            diff_pixel = (((-diff_lonlat / np.pi + 0.5) * pano_shape[0]) - h / 2).astype(np.int16)
            if abs(diff_pixel) > (h / 2 - h_pp / 2):
                diff_pixel = int(np.sign(diff_pixel) * (h / 2 - h_pp / 2))
                diff_lonlat = ((diff_pixel + h / 2) / pano_shape[0] - 0.5) * (-np.pi)
            equi_img = np.roll(equi_img, diff_pixel, axis=1)
            boundary[0, eval_c_idx] -= diff_lonlat
        
    elif gt_type == 2:
        f_idx = np.where((boundary[1] == thres) | (boundary[1] < thres))[0]
        if len(f_idx) > 0:
            diff_lonlat = min(boundary[1]) - thres
            diff_pixel = (((-diff_lonlat / np.pi + 0.5) * pano_shape[0]) - h / 2).astype(np.int16)
            if abs(diff_pixel) > (h / 2 - h_pp / 2):
                diff_pixel = int(np.sign(diff_pixel) * (h / 2 - h_pp / 2))
                diff_lonlat = ((diff_pixel + h / 2) / pano_shape[0] - 0.5) * (-np.pi)
            equi_img = np.roll(equi_img, diff_pixel, axis=1)
            boundary[1, eval_f_idx] -= diff_lonlat
        
    elif gt_type == 0:
        middle_lonlat = (boundary[0].max() + boundary[1].min()) / 2
        middle_pixel = (((-middle_lonlat / np.pi + 0.5) * pano_shape[0]) - h / 2).astype(np.int16)
        equi_img = np.roll(equi_img, middle_pixel, axis=1)
        boundary[0, eval_c_idx] -= middle_lonlat
        boundary[1, eval_f_idx] -= middle_lonlat
        diff_lonlat = middle_lonlat
        diff_pixel = middle_pixel
    else:
        raise ValueError('gt_type should be ceiling, floor or both, 0 for both, 1 for ceiling, 2 for floor')
    
    return equi_img, boundary, diff_lonlat, diff_pixel

def roll_img_boundary_predict(equi_img, boundary, eval_range, gt_type, pred_horizon, thres=60, pano_shape=(512,1024), pp_shape=(256,256)):
    '''
    Input:
        equi_img: equirectangular image (3, h, w)
        boundary: (2, w)
        eval_range: (2, w)
        gt_type: 0 for both, 1 for ceiling, 2 for floor
        thres: threshold of boundary
        pano_shape: (h, w)
    output:
        equi_img: rolled equirectangular image
        boundary: rolled boundary
        diff_pixel: number of pixels rolled
    '''
    h, _ = pano_shape
    h_pp, _ = pp_shape
    thres = np.deg2rad(thres)
    diff_lonlat = 0
    diff_pixel = 0
    eval_c_idx = np.where(eval_range[0] != 0)[0]
    eval_f_idx = np.where(eval_range[1] != 0)[0]
    if gt_type == 1:
        diff_pixel = -pred_horizon
        equi_img = np.roll(equi_img, diff_pixel, axis=1)
        diff_lonlat = ((diff_pixel + h / 2) / pano_shape[0] - 0.5) * (-np.pi)
        boundary[0, eval_c_idx] -= diff_lonlat
    elif gt_type == 2:
        diff_pixel = -pred_horizon
        equi_img = np.roll(equi_img, diff_pixel, axis=1)
        diff_lonlat = ((diff_pixel + h / 2) / pano_shape[0] - 0.5) * (-np.pi)
        boundary[1, eval_f_idx] -= diff_lonlat
    elif gt_type == 0:
        middle_pixel = -pred_horizon
        equi_img = np.roll(equi_img, middle_pixel, axis=1)
        middle_lonlat = ((middle_pixel + h / 2) / pano_shape[0] - 0.5) * (-np.pi)
        boundary[0, eval_c_idx] -= middle_lonlat
        boundary[1, eval_f_idx] -= middle_lonlat
        diff_lonlat = middle_lonlat
        diff_pixel = middle_pixel
    else:
        raise ValueError('gt_type should be ceiling, floor or both, 0 for both, 1 for ceiling, 2 for floor')
    
    return equi_img, boundary, diff_lonlat, diff_pixel

def boundary2depth_ceilingfloor(boundary, gt_type, eval_range, ch=1, thres=0.01):
    '''
    calculate horizon depth
    input:
        boundary: (2, 1024)
        gt_type: 0 for both, 1 for ceiling, 2 for floor
    output:
        depth: [ceiling_depth, floor_depth] (2,1024)
    '''
    # check boundary to avoid depth inf
    c_idx = np.where((boundary[0] > -thres) | (boundary[0] == -thres))[0]
    if len(c_idx) != 0:
        boundary[0, c_idx] = -thres
    f_idx = np.where(( boundary[1] < thres) | ( boundary[1] == thres))[0]
    if len(f_idx) != 0:
        boundary[1, f_idx] = thres
    
    bearing_floor = phi_coords2xyz(boundary[1])
    scale_floor = ch / bearing_floor[1, :]
    pcl_floor = scale_floor * bearing_floor
    floor_depth = np.linalg.norm(pcl_floor[(0, 2), :], axis=0)

    # calculate ceiling depth
    bearing_ceiling = phi_coords2xyz(boundary[0])
    scale_ceiling = -ch / bearing_ceiling[1, :]
    pcl_ceiling = scale_ceiling * bearing_ceiling
    ceiling_depth = np.linalg.norm(pcl_ceiling[(0, 2), :], axis=0)
    # merge depth
    depth = np.stack([ceiling_depth, floor_depth], axis=0)
   
    return depth

def boundary2depth_ceilingfloor_torch(boundary, gt_type, eval_range, ch=1, thres=0.01):
    '''
    calculate horizon depth
    input:
        boundary: (2, 1024)
        gt_type: 0 for both, 1 for ceiling, 2 for floor
    output:
        depth: [ceiling_depth, floor_depth] (2,1024)
    '''

    bearing_floor = phi_coords2xyz_torch(boundary[1])
    scale_floor = ch / bearing_floor[1, :]
    pcl_floor = scale_floor * bearing_floor
    floor_depth = torch.norm(pcl_floor[(0, 2), :], dim=0)

    # calculate ceiling depth
    bearing_ceiling = phi_coords2xyz_torch(boundary[0])
    scale_ceiling = -ch / bearing_ceiling[1, :]
    pcl_ceiling = scale_ceiling * bearing_ceiling
    ceiling_depth = torch.norm(pcl_ceiling[(0, 2), :], dim=0)

    # merge depth
    depth = torch.stack([ceiling_depth, floor_depth], dim=0)

    return depth

def cal_ratio(boundary, ch=1.6):
    '''
    calculate ratio of ceiling and floor boundary
    input:
        boundary: (2, 1024)
    output:
        ratio
    '''
    bearing_floor = phi_coords2xyz(boundary[1])
    scale_floor = ch / bearing_floor[1, :]
    pcl_floor = scale_floor * bearing_floor

    bearing_ceiling = phi_coords2xyz(boundary[0])
    scale_ceiling = np.linalg.norm(pcl_floor[(0, 2), :], axis=0) / np.linalg.norm(bearing_ceiling[(0, 2), :], axis=0)
    pcl_ceiling = scale_ceiling * bearing_ceiling
    ratio = abs(pcl_ceiling[1, :].mean() - ch)

    return ratio

def pp_map2boundary(pp_map, shape=(1000,1000)):
    '''
    Input:
        pp_b_c: ceiling boundary map (256,256)
        pp_b_f: floor boundary map (256,256)
        pp_shape: (h, w)
        out_bound_val: value of boundary outside image plane
    '''
    boundary = np.zeros((2, shape[1]))
    boundary[0] = -1
    boundary[1] = 1.1*shape[0]
    # 0: wall, 1: ceiling, 2: floor
    for i in range(shape[1]):
        c_index = np.where(pp_map[:, i] == 1)[0]
        f_index = np.where(pp_map[:, i] == 2)[0]
        if len(c_index) > 0:
            boundary[0, i] = c_index.max()
        if len(f_index) > 0:
            boundary[1, i] = f_index.min()
    
    return boundary

def get_ly_segmentation(xyz, shape=(100, 100)):
    
    ceiling_xyz = xyz[0]/xyz[0][2, :] if xyz[0].size > 0 else None
    floor_xyz = xyz[1]/xyz[1][2, :] if xyz[1].size > 0 else None

    # pp grid plane
    h, w = shape
    grid = np.zeros((h+2, w+2))
    
    # xyz ceiling 
    if ceiling_xyz is not None:
        u = np.clip(w/2 * ceiling_xyz[0] + w/2 + 0.5 , 0, w-1).astype(np.int32)
        v = np.clip(h/2 * ceiling_xyz[1] + h/2 + 0.5, 0, h-1).astype(np.int32)
        
        u[0]= 0
        u[-1] = w-1
        v[0] = 0
        v[-1] = 0
        
        cv2.fillPoly(grid, [np.vstack((u,v)).T], color=1)
        
    # xyz floor 
    if floor_xyz is not None:
        u = np.clip(w/2 * floor_xyz[0] + w/2 + 0.5 , 0, w-1).astype(np.int32)
        v = np.clip(h/2 * floor_xyz[1] + h/2 + 0.5, 0, h-1).astype(np.int32)
        
        u[0]= 0
        u[-1] = w-1
        v[0] = h-1
        v[-1] = h-1
        cv2.fillPoly(grid, [np.vstack((u,v)).T], color=2)
        
    return grid[1:-1, 1:-1]

def gen_pp_boundary_map(gt, est, gt_range, est_range, gt_v_shifting=0, shape=(1000, 1000)):
    gt_v_shifting = np.pi*gt_v_shifting/512
    xyz_gt = [phi_coords2xyz(phi_coords-gt_v_shifting)[:,  m.astype(np.bool8)] for phi_coords, m in zip(gt, gt_range)]
    xyz_est = [phi_coords2xyz(phi_coords-gt_v_shifting)[:,  m.astype(np.bool8)] for phi_coords, m in zip(est, est_range)]
    gt_map = get_ly_segmentation(xyz=xyz_gt, shape=shape)
    st_map = get_ly_segmentation(xyz=xyz_est, shape=shape)

    return gt_map, st_map
