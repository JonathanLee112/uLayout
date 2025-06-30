from shapely.geometry import Polygon
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from utils.grad import get_all

def phi_coords2xyz(phi_coords, theta_coords=None):
    # assert phi_coords.size == 1024, "phi_coords must be a 1024 array"
    if theta_coords is None:
        # due to circularity, we cannot define from -pi to pi
        # -pi and pi are the same point
        # we need to define from -pi to pi - 2pi/1024
        columns = phi_coords.size
        theta_coords = np.linspace(-np.pi, np.pi - 2 * np.pi / columns, columns)

    x = np.cos(phi_coords) * np.sin(theta_coords)
    y = np.sin(phi_coords)
    z = np.cos(phi_coords) * np.cos(theta_coords)

    return np.vstack((x, y, z))

def phi_coords2xyz_torch(phi_coords):
    W = phi_coords.size(dim=0)
    u = torch.linspace(0, W - 1, W).to(phi_coords.device)
    theta_coords = (2 * np.pi * u / W) - np.pi
    bearings_y = torch.sin(phi_coords)
    bearings_x = torch.cos(phi_coords) * torch.sin(theta_coords)
    bearings_z = torch.cos(phi_coords) * torch.cos(theta_coords)
    return torch.vstack((bearings_x, bearings_y, bearings_z))


def compute_L1_loss(y_est, y_ref):
        return F.l1_loss(y_est, y_ref)

def compute_L1_loss_range(y_est, y_ref, eval_range):
    '''
    input :
        y_est : (2, w) estimated boundary
        y_ref : (2, w) reference boundary
        eval_range : (2, w) pp evaluation range
    '''
    eval_c_idx = torch.where(eval_range[0,:] != 0)[0]
    eval_f_idx = torch.where(eval_range[1,:] != 0)[0]
    if len(eval_c_idx) == 0:
        c_loss = 0
    else:
        c_loss = F.l1_loss(y_est[0, eval_c_idx], y_ref[0, eval_c_idx])
    if len(eval_f_idx) == 0:
        f_loss = 0
    else:
        f_loss = F.l1_loss(y_est[1, eval_f_idx], y_ref[1, eval_f_idx])
    total_loss = c_loss + f_loss
    return total_loss

def compute_normal_gradient_loss(y_est, y_ref, coefficient=0.1):
    loss = nn.L1Loss()
    cos = nn.CosineSimilarity(dim=-1, eps=0)

    grad_conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0, bias=False, padding_mode='circular').to(y_est.device)
    grad_conv.weight = nn.Parameter(torch.tensor([[[1, 0, -1]]]).float().to(y_est.device))
    grad_conv.weight.requires_grad = False

    gt_direction, _, gt_angle_grad = get_all(y_ref[None,...], grad_conv)
    dt_direction, _, dt_angle_grad = get_all(y_est[None,...], grad_conv)

    normal_loss = (1 - cos(gt_direction, dt_direction)).mean()
    grad_loss = loss(gt_angle_grad, dt_angle_grad)

    return coefficient * (normal_loss + grad_loss)

def compute_normal_gradient_loss_range(y_est, y_ref, eval_range, coefficient=0.1):
    c_normal_loss = 0
    c_grad_loss = 0
    f_normal_loss = 0
    f_grad_loss = 0
    loss = nn.L1Loss()
    cos = nn.CosineSimilarity(dim=-1, eps=0)

    grad_conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0, bias=False, padding_mode='circular').to(y_est.device)
    grad_conv.weight = nn.Parameter(torch.tensor([[[1, 0, -1]]]).float().to(y_est.device))
    grad_conv.weight.requires_grad = False

    eval_c_idx = torch.where(eval_range[0,:] != 0)[0]
    eval_f_idx = torch.where(eval_range[1,:] != 0)[0]
    if len(eval_c_idx) > 0:
        gt_direction, _, gt_angle_grad = get_all(y_ref[0,eval_c_idx][None,...], grad_conv)
        dt_direction, _, dt_angle_grad = get_all(y_est[0,eval_c_idx][None,...], grad_conv)

        c_normal_loss = (1 - cos(gt_direction, dt_direction)).mean()
        c_grad_loss = loss(gt_angle_grad, dt_angle_grad)
    if len(eval_f_idx) > 0:
        gt_direction, _, gt_angle_grad = get_all(y_ref[1,eval_f_idx][None,...], grad_conv)
        dt_direction, _, dt_angle_grad = get_all(y_est[1,eval_f_idx][None,...], grad_conv)

        f_normal_loss = (1 - cos(gt_direction, dt_direction)).mean()
        f_grad_loss = loss(gt_angle_grad, dt_angle_grad)

    return coefficient * (c_normal_loss + c_grad_loss + f_normal_loss + f_grad_loss)
    
def compute_weighted_L1(y_est, y_ref, std, min_std=1E-2):
    return F.l1_loss(y_est/(std + min_std)**2, y_ref/(std + min_std)**2) 

def compute_weighted_L1_range(y_est, y_ref, std, eval_range, min_std=1E-2):
    '''
    input :
        y_est : (2, w) estimated boundary
        y_ref : (2, w) reference boundary
        std : (2, w) std of estimated boundary
        eval_range : (2, w) pp evaluation range
    '''
    eval_c_idx = torch.where(eval_range[0,:] != 0)[0]
    eval_f_idx = torch.where(eval_range[1,:] != 0)[0]
    if len(eval_c_idx) == 0:
        c_loss = 0
    else:
        c_loss = F.l1_loss(y_est[0,eval_c_idx]/(std[0,eval_c_idx] + min_std)**2, 
                            y_ref[0,eval_c_idx]/(std[0,eval_c_idx] + min_std)**2)
    if len(eval_f_idx) == 0:
        f_loss = 0
    else:
        f_loss = F.l1_loss(y_est[1,eval_f_idx]/(std[1,eval_f_idx] + min_std)**2, 
                            y_ref[1,eval_f_idx]/(std[1,eval_f_idx] + min_std)**2)
    total_loss = c_loss + f_loss
    return total_loss

def compute_L1_ceiling_loss(y_est, y_ref):
        y_est[1,:] = 0
        y_ref[1,:] = 0
        return F.l1_loss(y_est, y_ref) 

def compute_weighted_L1_ceiling(y_est, y_ref, std, min_std=1E-2):
        y_est[1,:] = 0
        y_ref[1,:] = 0
        return F.l1_loss(y_est/(std + min_std)**2, y_ref/(std + min_std)**2)

def compute_L1_floor_loss(y_est, y_ref):
        y_est[0,:] = 0
        y_ref[0,:] = 0
        return F.l1_loss(y_est, y_ref)

def compute_weighted_L1_floor(y_est, y_ref, std, min_std=1E-2):
        y_est[0,:] = 0
        y_ref[0,:] = 0
        return F.l1_loss(y_est/(std + min_std)**2, y_ref/(std + min_std)**2)

def find_predict_range(img, gt_boundary, boundary, u_range, gt_type):
    '''
    input : 
        img : (3, H, W)
        boundary : (2, W)
        u_range : (2, W)
        gt_type : 0 for both, 1 for ceiling, 2 for floor
    output :
        est_eval_range : (2,W) pp prediction range
    '''
    est_eval_range = np.zeros(boundary.shape).astype(np.int16)
    u_index = np.where(u_range != 0)[0]
    img = img.transpose(1,2,0).sum(axis=-1)
    boundary_pixel = ((boundary / np.pi + 0.5) * img.shape[0] - 0.5).astype(np.int16)
    # from u_index[0] to u_index[1]+1
    for j in range(u_index[0], u_index[-1]+1):
        index = np.where(img[:,j] != 0)[0]
        if len(index) > 0:
            if min(index) <= boundary_pixel[0,j]:
                est_eval_range[0,j] = 1
            if max(index) >= boundary_pixel[1,j]:
                est_eval_range[1,j] = 1

    if gt_type == 1:
        est_eval_range[1,:] = 0
    elif gt_type == 2:
        est_eval_range[0,:] = 0

    return est_eval_range.astype(np.int16), boundary


def eval_2d3d_iuo_from_tensors(est_bon, gt_bon, losses, gt_pp_range, pred_pp_range, u_range, ch=1, corner=None, ratio=1):
    est_bon_c = est_bon[:, 0, :].squeeze()
    est_bon_f = est_bon[:, 1, :].squeeze()
    gt_bon_c = gt_bon[:, 0, :].squeeze()
    gt_bon_f = gt_bon[:, 1, :].squeeze()

    est_bearing_ceiling = phi_coords2xyz(est_bon_c)
    est_bearing_floor = phi_coords2xyz(est_bon_f)
    gt_bearing_ceiling = phi_coords2xyz(gt_bon_c)
    gt_bearing_floor = phi_coords2xyz(gt_bon_f)

    # extract valid gt and pred range
    gt_c_idx = np.where(gt_pp_range[0,:] != 0)[0]
    gt_f_idx = np.where(gt_pp_range[1,:] != 0)[0]
    pred_c_idx = np.where(pred_pp_range[0,:] != 0)[0]
    pred_f_idx = np.where(pred_pp_range[1,:] != 0)[0]
    u_range_idx = np.where(u_range != 0)[0]

    if check_ceiling(gt_bearing_ceiling[1,:]) or check_floor(gt_bearing_floor[1,:]):
        diff = cal_diff(gt_bon_c, gt_bon_f, gt_c_idx, gt_f_idx)
        gt_bon_c, gt_bon_f = shift_boundary(gt_bon_c, gt_bon_f, gt_c_idx, gt_f_idx, diff)
        est_bon_c, est_bon_f = shift_boundary(est_bon_c, est_bon_f, pred_c_idx, pred_f_idx, diff)
        est_bearing_ceiling = phi_coords2xyz(est_bon_c)
        est_bearing_floor = phi_coords2xyz(est_bon_f)
        gt_bearing_ceiling = phi_coords2xyz(gt_bon_c)
        gt_bearing_floor = phi_coords2xyz(gt_bon_f)

    if len(gt_c_idx) == gt_pp_range.shape[-1] and len(gt_f_idx) == gt_pp_range.shape[-1]:
        iou2d, iou3d = get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor, est_bearing_ceiling, gt_bearing_ceiling)
        
        losses["2DIoU_pano"].append(iou2d)
        losses["3DIoU_pano"].append(iou3d)
    else:
        if len(gt_f_idx) == 0:
            iou2d = 0
        else:
            iou2d = get_pp_2d_iou(ch, est_bearing_floor[:,pred_f_idx],
                                    gt_bearing_floor[:,gt_f_idx])
            losses["2DIoU_pp_floor"].append(iou2d)
        if len(gt_c_idx) == 0:
            iou3d = 0
        else:
            iou3d = get_pp_2d_iou(-ch, est_bearing_ceiling[:,pred_c_idx],
                                    gt_bearing_ceiling[:,gt_c_idx])
            losses["2DIoU_pp_ceiling"].append(iou3d)

def eval_2d3d_iuo(phi_coords_est, phi_coords_gt_bon, ch=1):
    est_bearing_ceiling = phi_coords2xyz(phi_coords_est[0])
    est_bearing_floor = phi_coords2xyz(phi_coords_est[1])
    gt_bearing_ceiling = phi_coords2xyz(phi_coords_gt_bon[0])
    gt_bearing_floor = phi_coords2xyz(phi_coords_gt_bon[1])
    return get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor, est_bearing_ceiling, gt_bearing_ceiling)
    # Project bearings into a xz plane, ch: camera height

def get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor, est_bearing_ceiling, gt_bearing_ceiling):
    est_scale_floor = ch / est_bearing_floor[1, :]
    est_pcl_floor = est_scale_floor * est_bearing_floor

    gt_scale_floor = ch / gt_bearing_floor[1, :]
    gt_pcl_floor = gt_scale_floor * gt_bearing_floor

    est_scale_ceiling = np.linalg.norm(est_pcl_floor[(0, 2), :], axis=0) / np.linalg.norm(est_bearing_ceiling[(0, 2), :], axis=0)
    est_pcl_ceiling = est_scale_ceiling * est_bearing_ceiling
    est_h = abs(est_pcl_ceiling[1, :].mean() - ch)

    gt_scale_ceiling = np.linalg.norm(gt_pcl_floor[(0, 2), :], axis=0) / np.linalg.norm(gt_bearing_ceiling[(0, 2), :], axis=0)
    gt_pcl_ceiling = gt_scale_ceiling * gt_bearing_ceiling
    gt_h = abs(gt_pcl_ceiling[1, :].mean() - ch)

    try:
        est_poly = Polygon(zip(est_pcl_floor[0], est_pcl_floor[2]))
        gt_poly = Polygon(zip(gt_pcl_floor[0], gt_pcl_floor[2]))

        if not gt_poly.is_valid:
            print("[ERROR] Skip ground truth invalid")
            return -1, -1

        # 2D IoU
        try:
            area_dt = est_poly.area
            area_gt = gt_poly.area
            area_inter = est_poly.intersection(gt_poly).area
            iou2d = area_inter / (area_gt + area_dt - area_inter)
        except:
            iou2d = 0

        # 3D IoU
        try:
            area3d_inter = area_inter * min(est_h, gt_h)
            area3d_pred = area_dt * est_h
            area3d_gt = area_gt * gt_h
            iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
        except:
            iou3d = 0
    except:
        iou2d = 0
        iou3d = 0

    return iou2d, iou3d

def get_2d3d_iou_corner(ch, est_bearing_floor, est_bearing_ceiling, gt_floor_corner, ratio):
    est_scale_floor = ch / est_bearing_floor[1, :]
    est_pcl_floor = est_scale_floor * est_bearing_floor

    est_scale_ceiling = np.linalg.norm(est_pcl_floor[(0, 2), :], axis=0) / np.linalg.norm(est_bearing_ceiling[(0, 2), :], axis=0)
    est_pcl_ceiling = est_scale_ceiling * est_bearing_ceiling
    est_h = abs(est_pcl_ceiling[1, :].mean() - ch)

    gt_h = ratio * ch + ch
    

    try:
        est_poly = Polygon(zip(est_pcl_floor[0], est_pcl_floor[2]))
        gt_poly = Polygon(zip(gt_floor_corner[0], gt_floor_corner[2]))

        if not gt_poly.is_valid:
            print("[ERROR] Skip ground truth invalid")
            return -1, -1

        # 2D IoU
        try:
            area_dt = est_poly.area
            area_gt = gt_poly.area
            area_inter = est_poly.intersection(gt_poly).area
            iou2d = area_inter / (area_gt + area_dt - area_inter)
        except:
            iou2d = 0

        # 3D IoU
        try:
            area3d_inter = area_inter * min(est_h, gt_h)
            area3d_pred = area_dt * est_h
            area3d_gt = area_gt * gt_h
            iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
        except:
            iou3d = 0
    except:
        iou2d = 0
        iou3d = 0

    return iou2d, iou3d

def get_pp_2d_iou(ch, est_bearing, gt_bearing, thres=0.01):
    if ch > 0:
        gt_f_idx = check_floor(gt_bearing[1,:], thres=thres, format='idx')
        if len(gt_f_idx) > 0:
            gt_bearing[1,gt_f_idx] = thres
        est_f_idx = check_floor(est_bearing[1,:], thres=thres, format='idx')
        if len(est_f_idx) > 0:
            est_bearing[1,est_f_idx] = thres
    else:
        gt_c_idx = check_ceiling(gt_bearing[1,:], thres=-thres, format='idx')
        if len(gt_c_idx) > 0:
            gt_bearing[1,gt_c_idx] = -thres
        est_c_idx = check_ceiling(est_bearing[1,:], thres=-thres, format='idx')
        if len(est_c_idx) > 0:
            est_bearing[1,est_c_idx] = -thres

    est_scale = ch / est_bearing[1,:]
    est_pcl = est_scale * est_bearing

    gt_scale = ch / gt_bearing[1,:]
    gt_pcl = gt_scale * gt_bearing

    # camera origin with floor height and ceiling height
    c_origin = np.array([0, ch, 0])[None,...].transpose()

    est_pcl = np.column_stack((c_origin, est_pcl, c_origin))
    gt_pcl = np.column_stack((c_origin, gt_pcl, c_origin))

    try:
        est_poly = Polygon(zip(est_pcl[0], est_pcl[2]))
        gt_poly = Polygon(zip(gt_pcl[0], gt_pcl[2]))

        if not gt_poly.is_valid:
            print("[ERROR] Skip ground truth invalid")
            return -1
        
        try:
            area_dt = est_poly.area
            area_gt = gt_poly.area
            area_inter = est_poly.intersection(gt_poly).area
            iou2d = area_inter / (area_gt + area_dt - area_inter)

        except:
            iou2d = 0
    except:
        iou2d = 0
    
    return iou2d

def check_ceiling(c_y, thres=0, format='bool'):
    c_idx = np.where((c_y > thres) | (c_y == thres))[0]
    if format == 'bool':
        if len(c_idx) == 0:
            return False
        else:
            return True
    elif format == 'idx':
        return c_idx
    else:
        raise NotImplementedError("format should be 'bool' or 'idx'")
    
def check_floor(f_y, thres=0, format='bool'):
    f_idx = np.where((f_y < thres) | (f_y == thres))[0]
    if format == 'bool':
        if len(f_idx) == 0:
            return False
        else:
            return True
    elif format == 'idx':
        return f_idx
    else:
        raise NotImplementedError("format should be 'bool' or 'idx'")
    
def cal_diff(c_boundary, f_boundary, eval_c_idx, eval_f_idx):
    if len(eval_c_idx) > len(eval_f_idx):
        diff = np.mean(c_boundary[eval_f_idx] - f_boundary[eval_f_idx]) / 2
    else:
        diff = np.mean(c_boundary[eval_c_idx] - f_boundary[eval_c_idx]) / 2

    return diff

def shift_boundary(c_boundary, f_boundary, eval_c_idx, eval_f_idx, diff):
    c_boundary[eval_c_idx] -= diff
    f_boundary[eval_f_idx] -= diff

    return c_boundary, f_boundary