# Description: Utility functions for preprocessing
import os
import json
import pdb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from skimage import transform
from utils import postproc3d

plt.switch_backend('agg')

def extract_edgelets(img_id, img, out_dir, QUANT, SKIP_PAD, save=False):
    edgelets = postproc3d.compute_edgelets(img[SKIP_PAD:-SKIP_PAD, SKIP_PAD:-SKIP_PAD], quant=QUANT)
    locations, directions, strength = edgelets
    locations += SKIP_PAD
    edgelets = np.concatenate([locations, directions, strength[:, None]], -1)

    if save:
        out_dir = os.path.join(out_dir, 'edgelets_0.3')
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, img_id), edgelets)
    
    return edgelets

def prepare_vp(img_id, img, c_c, c_f, c_cw, c_fw, out_dir, save=False, vis=False):
    # ceiling_corner = kp[lsun_form['ceiling']].astype(np.float32)
    # floor_corner = kp[lsun_form['floor']].astype(np.float32)
    # ceiling_wall = kp[lsun_form['ceiling-wall']].astype(np.float32)
    # floor_wall = kp[lsun_form['floor-wall']].astype(np.float32)

    ceiling_corner = c_c.astype(np.float32)
    floor_corner = c_f.astype(np.float32)
    ceiling_wall = c_cw.astype(np.float32)
    floor_wall = c_fw.astype(np.float32)

    z_l = [np.cross([*ceiling_wall[i], 1], [*floor_wall[i], 1])
           for i in range(len(ceiling_wall))]
    z_pt = [[ceiling_wall[i], floor_wall[i]] for i in range(len(ceiling_wall))]
    x_l, y_l = [], []
    x_pt, y_pt = [], []
    for i in range(1, len(ceiling_corner)):
        l = np.cross([*ceiling_corner[i-1], 1], [*ceiling_corner[i], 1])
        if i % 2 == 1:
            x_l.append(l)
            x_pt.append([ceiling_corner[i-1], ceiling_corner[i]])
        else:
            y_l.append(l)
            y_pt.append([ceiling_corner[i-1], ceiling_corner[i]])
    for i in range(1, len(floor_corner)):
        l = np.cross([*floor_corner[i-1], 1], [*floor_corner[i], 1])
        if i % 2 == 1:
            x_l.append(l)
            x_pt.append([floor_corner[i-1], floor_corner[i]])
        else:
            y_l.append(l)
            y_pt.append([floor_corner[i-1], floor_corner[i]])

    z_info = postproc3d.info_from_aligned_layout(z_l, z_pt)
    x_info = postproc3d.info_from_aligned_layout(x_l, x_pt)
    y_info = postproc3d.info_from_aligned_layout(y_l, y_pt)

    z_res, x_res, y_res, edgelets = postproc3d.detect_3vp_from_xyzinfo(img, z_info, x_info, y_info)
    
    if vis:
        vpz = z_res['vp'].astype(np.float32)
        vpx = x_res['vp'].astype(np.float32)
        vpy = y_res['vp'].astype(np.float32)
        vpz /= vpz[-1]
        vpx /= vpx[-1]
        vpy /= vpy[-1]

        vis_dir = os.path.join(out_dir, 'debug_lsun_vp_vis')
        os.makedirs(vis_dir, exist_ok=True)
        plt.xlim(0, 512)
        plt.ylim(512, 0)
        plt.imshow(img)
        for location, direction, strength in zip(*edgelets):
            x1, y1 = location - direction * strength / 2
            x2, y2 = location + direction * strength / 2
            plt.plot([x1, x2], [y1, y2], 'y--')
        for location, direction, strength in zip(*z_res['edgelets']):
            x1, y1 = location - direction * strength / 2
            x2, y2 = location + direction * strength / 2
            plt.plot([location[0], vpz[0]], [location[1], vpz[1]], 'r:', alpha=0.3)
            plt.plot([x1, x2], [y1, y2], 'r.-')
        for location, direction, strength in zip(*x_res['edgelets']):
            x1, y1 = location - direction * strength / 2
            x2, y2 = location + direction * strength / 2
            plt.plot([location[0], vpx[0]], [location[1], vpx[1]], 'g:', alpha=0.3)
            plt.plot([x1, x2], [y1, y2], 'g.-')
        for location, direction, strength in zip(*y_res['edgelets']):
            x1, y1 = location - direction * strength / 2
            x2, y2 = location + direction * strength / 2
            plt.plot([location[0], vpy[0]], [location[1], vpy[1]], 'b:', alpha=0.3)
            plt.plot([x1, x2], [y1, y2], 'b.-')
        plt.savefig('%s/%s' % (vis_dir, img_id))
        plt.clf()

    if save:
        out_dir = os.path.join(out_dir, 'debug_lsun_vp')
        os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_dir}/{img_id}.json', 'w') as f:
            # print(z_res['vp'] / z_res['vp'][-1])
            # print(x_res['vp'] / x_res['vp'][-1])
            # print(y_res['vp'] / y_res['vp'][-1])
            json.dump({
                'z_vp': tuple(float(v) for v in z_res['vp']),
                'x_vp': tuple(float(v) for v in x_res['vp']),
                'y_vp': tuple(float(v) for v in y_res['vp']),
            }, f)

    return z_res, x_res, y_res

def extract_z_vp(img_id, img, c_cw, c_fw, out_dir,edgelets,
                VP_THRES, DEG_THRES, RANSAC_IT, save=False, vis=False):
    DEG_THRES2 = np.deg2rad(DEG_THRES)

    # Filter edgelets
    edgelets = (edgelets[:, [0, 1]], edgelets[:, [2, 3]], edgelets[:, 4])
    rad = np.arctan2(edgelets[1][:, 1], edgelets[1][:, 0])
    candidates = (
        ((DEG_THRES2 < rad) & (rad < np.pi - DEG_THRES2)) |\
        ((-(np.pi - DEG_THRES2) < rad) & (rad < -DEG_THRES2))
    )
    edgelets = tuple(v[candidates] for v in edgelets)
    vpz = postproc3d.ransac_vanishing_point(edgelets, VP_THRES, RANSAC_IT)
    inliers = postproc3d.detect_inliers(vpz, edgelets, VP_THRES)
    z_edgelets = tuple(d[inliers] for d in edgelets)
    vpz = postproc3d.refine_model(vpz, z_edgelets)
    vpz = vpz / (vpz[-1] if abs(vpz[-1]) > 1e-9 else 1e-9)

    # Check for fallback
    deg = np.arctan2(vpz[1], vpz[0]) * 180 / np.pi
    if (-DEG_THRES < deg and deg < DEG_THRES) or (deg > 180-DEG_THRES) or (deg < -180+DEG_THRES):
        inliers = postproc3d.detect_inliers(vpz, edgelets, VP_THRES)
        edgelets2 = tuple(d[~inliers] for d in edgelets)
        vpz2 = postproc3d.ransac_vanishing_point(edgelets2, VP_THRES, RANSAC_IT)
        vpz2 = vpz2 / (vpz2[-1] if abs(vpz2[-1]) > 1e-9 else 1e-9)
        vpz = vpz2
        # print(f'fallback {img_id}')

    if save:
        out_dir = os.path.join(out_dir, 'debug_zvp_0.3')
        os.makedirs(out_dir, exist_ok=True)
        with open('%s/%s' % (out_dir, img_id + '.json'), 'w') as f:
            json.dump({
                'z_vp': tuple(float(v) for v in vpz),
                'img_w': 512,
                'img_h': 512,
            }, f)

    inliers = postproc3d.detect_inliers(vpz, edgelets, VP_THRES)
    z_edgelets = tuple(d[inliers] for d in edgelets)
    if len(z_edgelets) < len(edgelets) * 0.3:
        print('plz check', img_id)

    if vis:
        vis_dir = os.path.join(out_dir, 'debug_zvp_0.3_vis')
        os.makedirs(out_dir, exist_ok=True)
        # ceiling_wall = kp[lsun_form['ceiling-wall']].astype(np.float32)
        # floor_wall = kp[lsun_form['floor-wall']].astype(np.float32)
        ceiling_wall = c_cw.astype(np.float32)
        floor_wall = c_fw.astype(np.float32)
        plt.xlim(0, 512)
        plt.ylim(512, 0)
        plt.imshow(img)
        for location, direction, strength in zip(*edgelets):
            x1, y1 = location - direction * strength / 2
            x2, y2 = location + direction * strength / 2
            plt.plot([x1, x2], [y1, y2], 'y--')
        for location, direction, strength in zip(*z_edgelets):
            x1, y1 = location - direction * strength / 2
            x2, y2 = location + direction * strength / 2
            plt.plot([location[0], vpz[0]], [location[1], vpz[1]], 'r:', alpha=0.3)
            plt.plot([x1, x2], [y1, y2], 'r.-')
        for i in range(len(ceiling_wall)):
            cx = (ceiling_wall[i][0] + floor_wall[i][0]) / 2
            cy = (ceiling_wall[i][1] + floor_wall[i][1]) / 2
            plt.plot([cx, vpz[0]], [cy, vpz[1]], 'g--')
            plt.plot([ceiling_wall[i][0], floor_wall[i][0]], [ceiling_wall[i][1], floor_wall[i][1]], 'go-')
        plt.axis('off')

        rot = np.arctan2(vpz[1] - img.shape[0] / 2, vpz[0] - img.shape[1] / 2) * 180 / np.pi
        if rot < 0:
            rot = abs(rot + 90)
        else:
            rot = abs(rot - 90)
        plt.savefig('%s/%.2f_%s.jpg' % (vis_dir, rot, img_id), bbox_inches='tight')
        plt.clf()
    
    return vpz

def compute_homography_and_warp(image, vp1, vp2, clip=True, clip_factor=3):
    """Compute homography from vanishing points and warp the image.
    It is assumed that vp1 and vp2 correspond to horizontal and vertical
    directions, although the order is not assumed.
    Firstly, projective transform is computed to make the vanishing points go
    to infinty so that we have a fronto parellel view. Then,Computes affine
    transfom  to make axes corresponding to vanishing points orthogonal.
    Finally, Image is translated so that the image is not missed. Note that
    this image can be very large. `clip` is provided to deal with this.
    Parameters
    ----------
    image: ndarray
        Image which has to be wrapped.
    vp1: ndarray of shape (3, )
        First vanishing point in homogenous coordinate system.
    vp2: ndarray of shape (3, )
        Second vanishing point in homogenous coordinate system.
    clip: bool, optional
        If True, image is clipped to clip_factor.
    clip_factor: float, optional
        Proportion of image in multiples of image size to be retained if gone
        out of bounds after homography.
    Returns
    -------
    warped_img: ndarray
        Image warped using homography as described above.
    """
    # Find Projective Transform
    vanishing_line = np.cross(vp1, vp2)
    H = np.eye(3)
    H[2] = vanishing_line / vanishing_line[2]
    H = H / H[2, 2]

    # Find directions corresponding to vanishing points
    v_post1 = np.dot(H, vp1)
    v_post2 = np.dot(H, vp2)
    v_post1 = v_post1 / np.sqrt(v_post1[0]**2 + v_post1[1]**2)
    v_post2 = v_post2 / np.sqrt(v_post2[0]**2 + v_post2[1]**2)

    directions = np.array([[v_post1[0], -v_post1[0], v_post2[0], -v_post2[0]],
                           [v_post1[1], -v_post1[1], v_post2[1], -v_post2[1]]])

    thetas = np.arctan2(directions[0], directions[1])

    # Find direction closest to horizontal axis
    h_ind = np.argmin(np.abs(thetas))

    # Find positve angle among the rest for the vertical axis
    if h_ind // 2 == 0:
        v_ind = 2 + np.argmax([thetas[2], thetas[3]])
    else:
        v_ind = np.argmax([thetas[2], thetas[3]])

    A1 = np.array([[directions[0, v_ind], directions[0, h_ind], 0],
                   [directions[1, v_ind], directions[1, h_ind], 0],
                   [0, 0, 1]])
    # Might be a reflection. If so, remove reflection.
    if np.linalg.det(A1) < 0:
        A1[:, 0] = -A1[:, 0]

    A = np.linalg.inv(A1)

    # Translate so that whole of the image is covered
    inter_matrix = np.dot(A, H)

    cords = np.dot(inter_matrix, [[0, 0, image.shape[1], image.shape[1]],
                                  [0, image.shape[0], 0, image.shape[0]],
                                  [1, 1, 1, 1]])
    cords = cords[:2] / cords[2]

    tx = min(0, cords[0].min())
    ty = min(0, cords[1].min())

    max_x = cords[0].max() - tx
    max_y = cords[1].max() - ty

    if clip:
        # These might be too large. Clip them.
        max_offset = max(image.shape) * clip_factor / 2
        tx = max(tx, -max_offset)
        ty = max(ty, -max_offset)

        max_x = min(max_x, -tx + max_offset)
        max_y = min(max_y, -ty + max_offset)

    max_x = int(max_x)
    max_y = int(max_y)

    T = np.array([[1, 0, -tx],
                  [0, 1, -ty],
                  [0, 0, 1]])

    final_homography = np.dot(T, inter_matrix)

    warped_img = transform.warp(image,
                                inverse_map=np.linalg.inv(final_homography),
                                output_shape=(max_y, max_x))
    return warped_img, final_homography

def align_by_vpz(img_id, img, c_c, c_f, c_cw, c_fw, out_dir, data_mode, vpz, vis=False):
    ceiling_corner = c_c.astype(np.float32)
    floor_corner = c_f.astype(np.float32)
    ceiling_wall = c_cw.astype(np.float32)
    floor_wall = c_fw.astype(np.float32)
    # Check for no rot fallback
    rot = np.arctan2(vpz[1] - img.shape[0] / 2, vpz[0] - img.shape[1] / 2) * 180 / np.pi
    if rot < 0:
        rot = abs(rot + 90)
    else:
        rot = abs(rot - 90)
    if rot > 20:
        vpz = np.array([img.shape[1] / 2, 1e9, 1], np.float32)

    # Align by vpz
    l = np.cross(vpz, np.array([(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2, 1]))
    horizon = np.array([l[0], l[1], 0])
    img_aligned, H = compute_homography_and_warp(img, vpz, horizon)
    img_aligned = (img_aligned * 255).astype(np.uint8)
    if abs((H @ vpz)[0]) > 1e-6:
        print('plz plz plz check', img_id)

    # Save to target
    output_img_dir = os.path.join(out_dir, data_mode, 'img_aligned')
    os.makedirs(output_img_dir, exist_ok=True)
    output_label_dir = os.path.join(out_dir, data_mode, 'cor_aligned')
    os.makedirs(output_label_dir, exist_ok=True)
    Image.fromarray(img_aligned).save(os.path.join(output_img_dir, img_id + '.png'))
    with open(os.path.join(output_label_dir, img_id + '.json'), 'w') as f:
        ori = {
            'hw': [int(img.shape[0]), int(img.shape[1])],
            'z_vp': tuple(float(v) for v in vpz),
            'ceiling_corner': [[float(c[0]), float(c[1])] for c in ceiling_corner],
            'floor_corner': [[float(c[0]), float(c[1])] for c in floor_corner],
            'ceiling_wall': [[float(c[0]), float(c[1])] for c in ceiling_wall],
            'floor_wall': [[float(c[0]), float(c[1])] for c in floor_wall],
        }
        new = {
            'hw': [int(img_aligned.shape[0]), int(img_aligned.shape[1])],
            'z_vp': [float(v) for v in (H @ vpz)],
            'ceiling_corner': [],
            'floor_corner': [],
            'ceiling_wall': [],
            'floor_wall': [],
        }
        for c in ceiling_corner:
            nc = H @ np.array([*c, 1])
            nc = nc / nc[-1]
            new['ceiling_corner'].append([float(nc[0]), float(nc[1])])
        for c in floor_corner:
            nc = H @ np.array([*c, 1])
            nc = nc / nc[-1]
            new['floor_corner'].append([float(nc[0]), float(nc[1])])
        for c in ceiling_wall:
            nc = H @ np.array([*c, 1])
            nc = nc / nc[-1]
            new['ceiling_wall'].append([float(nc[0]), float(nc[1])])
        for c in floor_wall:
            nc = H @ np.array([*c, 1])
            nc = nc / nc[-1]
            new['floor_wall'].append([float(nc[0]), float(nc[1])])

        json.dump({
            'H': [[float(v) for v in row] for row in H],
            'img_name':  img_id,
            'ori': ori,
            'new': new,
        }, f, indent=1)

    if vis:
        vis_dir = os.path.join(out_dir, 'paper_demo')
        os.makedirs(vis_dir, exist_ok=True)
        # Visualize original w/ edgelets
        edgelets = np.load(os.path.join('edgelets_0.3', img_id + '.npy'))
        edgelets = (edgelets[:, [0, 1]], edgelets[:, [2, 3]], edgelets[:, 4])
        inliers = postproc3d.detect_inliers(vpz, edgelets, 5)
        z_edgelets = tuple(d[inliers] for d in edgelets)
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)
        plt.imshow(img)
        for location, direction, strength in zip(*z_edgelets):
            x1, y1 = location - direction * strength / 2
            x2, y2 = location + direction * strength / 2
            plt.plot([location[0], vpz[0]], [location[1], vpz[1]], 'r:', alpha=0.3)
            plt.plot([x1, x2], [y1, y2], 'm.-', alpha=0.8)
        plt.axis('off')
        plt.savefig('%s/%s.ori.jpg' % (vis_dir, img_id), bbox_inches='tight')
        plt.clf()

        # Visualize now w/ edgelets
        plt.xlim(0, img_aligned.shape[1])
        plt.ylim(img_aligned.shape[0], 0)
        plt.imshow(img_aligned)

        Hvpz = H @ vpz
        Hvpz = Hvpz / (Hvpz[-1] if Hvpz[-1] > 1e-9 else 1e-9)
        for location, direction, strength in zip(*z_edgelets):
            x1, y1 = location - direction * strength / 2
            x2, y2 = location + direction * strength / 2
            location = H @ np.array([*location, 1])
            location = location[:2] / location[2]
            x1y1 = H @ np.array([x1, y1, 1])
            x1, y1 = x1y1[:2] / x1y1[2]
            x2y2 = H @ np.array([x2, y2, 1])
            x2, y2 = x2y2[:2] / x2y2[2]
            plt.plot([location[0], Hvpz[0]], [location[1], Hvpz[1]], 'r:', alpha=0.3)
            plt.plot([x1, x2], [y1, y2], 'm.-', alpha=0.8)
        plt.axis('off')
        plt.savefig('%s/%s.now.jpg' % (vis_dir, img_id), bbox_inches='tight')
        plt.clf()