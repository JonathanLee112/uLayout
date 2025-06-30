import os
import json
import numpy as np
import open3d as o3d
import yaml
import glob
import pdb
import sys
from PIL import Image
from scipy.signal import correlate2d
from scipy.ndimage import shift
from layout_3d_utils import np_coor2xy, np_coorx2u, np_coory2v, layout_2_depth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.mp3d_hn_ori_dataset import PanoCorBonDataset
from datasets.pano_st2d3d_dataset import PanoSt2D3DDataset


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_dir', 
                        required=True,
                        default='src/pano/pano_st2d3d',
                        help='dataset dir')
    parser.add_argument('--dataset', 
                        required=True,
                        default='pano',
                        choices=['mp3d', 'pano', 'st2d3d'],
                        help='mp3d or pano or st2d3d')
    parser.add_argument('--mode',
                        default='test',
                        help='train or val or test')
    parser.add_argument('--index',
                        default=0,
                        type=int,
                        help='index of the data')   
    parser.add_argument('--layout', 
                        required=True,
                        help='Txt or json file containing layout corners (cor_id)')
    parser.add_argument('--output_dir', 
                        default='output',
                        help='Output directory')
    parser.add_argument('--out')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling', action='store_true',
                        help='Skip rendering ceiling')
    parser.add_argument('--ignore_wall', action='store_true',
                        help='Skip rendering wall')
    parser.add_argument('--ignore_wireframe', action='store_true',
                        help='Skip rendering wireframe')
    args = parser.parse_args()

    if not args.out and not args.vis:
        print('You may want to export (via --out) or visualize (via --vis)')
        import sys; sys.exit()

    if args.dataset == 'mp3d':
        dataset = PanoCorBonDataset(args.dataset_dir, args.mode)
    elif args.dataset == 'pano':
        dataset = PanoSt2D3DDataset(args.dataset_dir, mode=args.mode, subset='pano')
    elif args.dataset == 'st2d3d':
        dataset = PanoSt2D3DDataset(args.dataset_dir, mode=args.mode, subset='st2d3d')
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    # open json file
    est_dir = os.path.join(args.layout, f'est_{args.index}.json')
    with open(est_dir, 'r') as f:
        est = json.load(f)

    # transfer the torch tensor to numpy array
    equirect_texture = dataset[args.index]['img'].numpy().transpose(1, 2, 0) * 255
    H, W = equirect_texture.shape[:2]
    # cor_id = dataset[args.index]['cor_id'].numpy()
    cor_id = np.array(est['corner'])
    est_layout = np.array(est['boundary'])

    # Convert corners to layout
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(est_layout, H, W, return_mask=True)
    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth * np.sin(vs)
    cs = depth * np.cos(vs)
    xs = cs * np.sin(us)
    ys = -cs * np.cos(us)

    # Aggregate mask
    mask = np.ones_like(floor_mask)
    if args.ignore_floor:
        mask &= ~floor_mask
    if args.ignore_ceiling:
        mask &= ~ceil_mask
    if args.ignore_wall:
        mask &= ~wall_mask

    # Prepare ply's points and faces
    xyzrgb = np.concatenate([
        xs[...,None], ys[...,None], zs[...,None],
        equirect_texture], -1)
    xyzrgb = np.concatenate([xyzrgb, xyzrgb[:,[0]]], 1)
    mask = np.concatenate([mask, mask[:,[0]]], 1)
    lo_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1]])
    up_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 1]])
    ma_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 0]])
    lo_mask = (correlate2d(mask, lo_tri_template, mode='same') == 3)
    up_mask = (correlate2d(mask, up_tri_template, mode='same') == 3)
    ma_mask = (correlate2d(mask, ma_tri_template, mode='same') == 3) & (~lo_mask) & (~up_mask)
    ref_mask = (
        lo_mask | (correlate2d(lo_mask, np.flip(lo_tri_template, (0,1)), mode='same') > 0) |\
        up_mask | (correlate2d(up_mask, np.flip(up_tri_template, (0,1)), mode='same') > 0) |\
        ma_mask | (correlate2d(ma_mask, np.flip(ma_tri_template, (0,1)), mode='same') > 0)
    )
    points = xyzrgb[ref_mask]

    ref_id = np.full(ref_mask.shape, -1, np.int32)
    ref_id[ref_mask] = np.arange(ref_mask.sum())
    faces_lo_tri = np.stack([
        ref_id[lo_mask],
        ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
    ], 1)
    faces_up_tri = np.stack([
        ref_id[up_mask],
        ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
        ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces_ma_tri = np.stack([
        ref_id[ma_mask],
        ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])

    # Dump results ply
    if args.out:
        ply_header = '\n'.join([
            'ply',
            'format ascii 1.0',
            f'element vertex {len(points):d}',
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            f'element face {len(faces):d}',
            'property list uchar int vertex_indices',
            'end_header',
        ])
        with open(args.out, 'w') as f:
            f.write(ply_header)
            f.write('\n')
            for x, y, z, r, g, b in points:
                f.write(f'{x:.2f} {y:.2f} {z:.2f} {r:.0f} {g:.0f} {b:.0f}\n')
            for i, j, k in faces:
                f.write(f'3 {i:d} {j:d} {k:d}\n')

    if args.vis:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points[:, :3])
        mesh.vertex_colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        draw_geometries = [mesh]

        # Show wireframe
        if not args.ignore_wireframe:
            # Convert cor_id to 3d xyz
            N = len(cor_id) // 2
            floor_z = -1.6
            floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
            c = np.sqrt((floor_xy**2).sum(1))
            v = np_coory2v(cor_id[0::2, 1], H)
            ceil_z = (c * np.tan(v)).mean()

            # Prepare wireframe in open3d
            assert N == len(floor_xy)
            wf_points = [[x, y, floor_z] for x, y in floor_xy] +\
                        [[x, y, ceil_z] for x, y in floor_xy]
            wf_lines = [[i, (i+1)%N] for i in range(N)] +\
                       [[i+N, (i+1)%N+N] for i in range(N)] +\
                       [[i, i+N] for i in range(N)]
            wf_colors = [[1, 0, 0] for i in range(len(wf_lines))]
            wf_line_set = o3d.geometry.LineSet()
            wf_line_set.points = o3d.utility.Vector3dVector(wf_points)
            wf_line_set.lines = o3d.utility.Vector2iVector(wf_lines)
            wf_line_set.colors = o3d.utility.Vector3dVector(wf_colors)
            draw_geometries.append(wf_line_set)
        
        # output_dir = os.path.join(args.output_dir, '3D_layout')
        # os.makedirs(output_dir, exist_ok=True)
        
        o3d.visualization.draw_geometries(draw_geometries, mesh_show_back_face=True)
