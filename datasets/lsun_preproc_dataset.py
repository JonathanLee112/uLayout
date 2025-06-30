import json
import pdb
import os
import numpy as np
import cv2
import torch
import torch.utils.data as data

from utils.Cube2Equirec import Cube2Equirec
from utils.pp_utils import  read_image, read_json, rolling_img_boundary, \
                            boundary2depth_ceilingfloor, roll_img_boundary_predict

class LSUNPreprocDataset(data.Dataset):
    def __init__(self, root_dir, mode, fix_shape, req_depth=False, pano_rotate_angle=30, out_bound_val=75):
        self.root_dir = root_dir
        self.mode = mode
        self.fix_shape = fix_shape
        self.pano_rotate_angle = pano_rotate_angle
        self.out_bound_val = out_bound_val
        self.req_depth = req_depth
        self.gamma = False
        self.flip = False
        self.h_roll = False
        self.random_vertical_shift = False

        assert mode in ('train', 'val', 'test'), 'mode only support : train, val, test'
        if self.mode == 'train':
            self.gamma = True
            self.flip = True
            #self.random_vertical_shift = True
            #self.h_roll = True

        self.gt_dir = os.path.join(self.root_dir, mode, 'boundary_aligned_pano')
        self.img_dir = os.path.join(self.root_dir, mode, 'img_aligned_pano')
        json_path = os.path.join(self.root_dir, mode, f"{mode}_scene_list.json")
        pred_horizon_path = os.path.join(self.root_dir, mode, f'{mode}_lsun_pred_horizon.json')
        with open(json_path, 'r') as f:
            # data_list contains three types of data: ceiling_floor, only_ceiling, only_floor
            data_lst = json.load(f)
        self.data_list = data_lst['ceiling_floor'] + data_lst['only_floor'] + data_lst['only_ceiling']
        
        # load pred horizon
        with open(pred_horizon_path, 'r') as f:
            self.pred_horizon = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_path = os.path.join(self.img_dir, f"{data}.png")
        gt_path = os.path.join(self.gt_dir, f"{data}.json")
        img = read_image(img_path).transpose(2, 0, 1)
        gt = read_json(gt_path)
        # get gt
        gt_type = gt["gt_type"]
        name = gt["name"]
        boundary = np.array(gt["boundary"]).astype(np.float32)
        
        # data augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 1:
                p = 1 / p
            img = img ** p
        if self.flip and np.random.randint(2) == 1:
            # flip
            img = img[:, :, ::-1]
            boundary = boundary[:, ::-1]
            #ceiling_corner, floor_corner = aug_flip(ceiling_corner, floor_corner)
            

        # random horizontal roll
        if self.h_roll:
            # img, boundary = aug_h_roll(img, boundary, self.fix_shape)
            shift = np.random.randint(0, img.shape[-1])
            img = np.roll(img, shift, axis=-1)
            boundary = np.roll(boundary, shift, axis=-1)
        # get eval range
        eval_c_idx = np.where(boundary[0] != (-1*np.deg2rad(self.out_bound_val).astype(np.float32)))[0]
        eval_f_idx = np.where(boundary[1] != np.deg2rad(self.out_bound_val).astype(np.float32))[0]
        eval_range = np.zeros((2,boundary.shape[-1])).astype(np.int16)
        eval_range[0, eval_c_idx] = 1
        eval_range[1, eval_f_idx] = 1
        # get v_shift
        if gt_type == 0:
            img, boundary, _, v_shift_pixel = rolling_img_boundary(img, boundary, eval_range, gt_type, thres=10)
        else:
            pred_horizon_pixel = int((self.pred_horizon[name][0] - 256) / 2)
            img, boundary, _, v_shift_pixel = roll_img_boundary_predict(img, boundary, eval_range, gt_type, pred_horizon_pixel, thres=60)
        # get depth
        depth = boundary2depth_ceilingfloor(boundary, gt_type, eval_range)
        depth = torch.FloatTensor(depth.copy())

        # get u_range
        u_range = np.zeros(boundary.shape[-1]).astype(np.int16)
        index = np.array(np.where(img != 0))
        u_range[np.unique(index[-1])] = 1
        std = np.ones_like(boundary)
        boundary_gpu = torch.FloatTensor(boundary.copy())
        u_range = torch.FloatTensor(u_range.copy())
        eval_range = torch.FloatTensor(eval_range.copy())
        gt_type = torch.FloatTensor([gt_type+1])
        img_gpu = torch.FloatTensor(img.copy())
        v_shift_pixel = torch.FloatTensor([v_shift_pixel])
        cor = np.zeros((50, 2)).astype(np.float32)
        cor = torch.FloatTensor(cor.copy())


        output = {
            'img_name': data,
            'img': img_gpu,
            'corner': cor.T,
            'boundary': boundary_gpu,
            'depth': depth,
            'u_range': u_range,
            'eval_range': eval_range,
            'gt_type': gt_type,
            'v_shift': v_shift_pixel,
        }

        return output


