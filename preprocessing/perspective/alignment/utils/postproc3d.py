import cv2
import numpy as np
import scipy
import pdb
from skimage import color


def lsdWrap(img, LSD=None, **kwargs):
    if LSD is None:
        LSD = cv2.createLineSegmentDetector(**kwargs)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lines, width, prec, nfa = LSD.detect(image=img)
    return lines.reshape(len(lines), 2, 2)


def p0p1_2_edgelets(p0, p1):
    p0 = np.array(p0)
    p1 = np.array(p1)
    locations = (p0 + p1) / 2
    directions = p1 - p0
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    strengths = np.linalg.norm(p1 - p0, axis=1)

    return locations, directions, strengths


def compute_edgelets(image, **kwargs):
    """Create edgelets as in the paper.

    Uses canny edge detection and then finds (small) lines using probabilstic
    hough transform as edgelets.

    Parameters
    ----------
    image: ndarray
        Image for which edgelets are to be computed.
    sigma: float
        Smoothing to be used for canny edge detection.

    Returns
    -------
    locations: ndarray of shape (n_edgelets, 2)
        Locations of each of the edgelets.
    directions: ndarray of shape (n_edgelets, 2)
        Direction of the edge (tangent) at each of the edgelet.
    strengths: ndarray of shape (n_edgelets,)
        Length of the line segments detected for the edgelet.
    """
    gray_img = color.rgb2gray(image)
    lsd_kwargs = {
        'refine' : cv2.LSD_REFINE_ADV,
        'quant' : 2,
    }
    lsd_kwargs.update(kwargs)
    lines = lsdWrap(image, cv2.createLineSegmentDetector(**lsd_kwargs))
    p0 = lines[:, 0]
    p1 = lines[:, 1]

    return p0p1_2_edgelets(p0, p1)


def line_intersect_from_p(vp1, vp2, p1, p2):
    l1 = np.cross(vp1, p1)
    l2 = np.cross(vp2, p2)
    pt = np.cross(l1, l2)
    pt = pt / pt[2]
    return pt


def infer_by_cross_ratio(vp, p0, pr, pt, l0r):
    assert len(vp) == 3 and len(p0) == 3 and \
           len(pr) == 3 and len(pt) == 3

    if ((pt - p0)**2).sum() < 1e-6:
        return 0
    if ((pt - pr)**2).sum() < 1e-6:
        return l0r

    vp = vp / vp[-1]
    p0 = p0 / p0[-1]
    pr = pr / pr[-1]
    pt = pt / pt[-1]
    cross_ratio_p = np.array([vp, p0, pr, pt])
    x = cross_ratio_p[:, 0]
    assert x[0] == x.max() or x[0] == x.min()
    reidx = np.argsort(cross_ratio_p[:, 0])
    if 0 == reidx[0]:
        reidx = reidx[::-1]
    else:
        assert 0 == reidx[-1]
    cross_ratio_p = cross_ratio_p[reidx]
    dist = scipy.spatial.distance.pdist(cross_ratio_p)
    dist = scipy.spatial.distance.squareform(dist)
    cr_x = dist[0, 2] * dist[1, 3]
    cr_y = dist[0, 3] * dist[1, 2]

    tp = tuple(reidx[:3])
    if tp == (1, 2, 3):
        return abs(cr_x * l0r / (cr_x - cr_y))
    elif tp == (1, 3, 2):
        return abs(l0r * (1 - cr_y / cr_x))
    elif tp == (2, 1, 3):
        return -abs(l0r * cr_y / (cr_x - cr_y))
    elif tp == (2, 3, 1):
        return abs(l0r * cr_y / cr_x)
    elif tp == (3, 1, 2):
        return -abs(l0r * (cr_x / cr_y - 1))
    elif tp == (3, 2, 1):
        return abs(l0r * cr_x / cr_y)
    else:
        raise Exception()


def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.

    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def compute_votes(edgelets, model, threshold_inlier):
    """Compute votes for each of the edgelet against a given vanishing point.

    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.

    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.

    """
    theta_thresh = threshold_inlier * np.pi / 180
    locations, directions, strengths = edgelets

    # First criterion: angle(vp to edgelets center, edgelets direction)
    le = np.cross(np.hstack([locations, np.ones(len(locations))[:, None]]), model)
    est_directions = np.concatenate([-le[:, [1]], le[:, [0]]], 1)

    ve = est_directions / np.linalg.norm(est_directions, axis=1, keepdims=True)
    vd = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    theta = np.abs(np.arccos(np.clip((ve * vd).sum(axis=1), -1, 1)))
    theta = np.minimum(theta, np.pi - theta)

    # Second criterion: dot of vp to edgelets two end points
    pa = locations - directions * strengths[:, None] / 2
    pb = locations + directions * strengths[:, None] / 2
    la = np.cross(np.hstack([pa, np.ones(len(locations))[:, None]]), model)
    lb = np.cross(np.hstack([pb, np.ones(len(locations))[:, None]]), model)
    va = np.concatenate([-la[:, [1]], la[:, [0]]], 1)
    vb = np.concatenate([-lb[:, [1]], lb[:, [0]]], 1)
    dot = (va * vb).sum(1)

    criterion1 = (theta < theta_thresh)
    criterion2 = (dot > 0)
    return (criterion1 & criterion2) * strengths**2 * (1 - theta / (np.pi / 2))


def scanline_vanishing_point(edgelets, base_edgelet, threshold_inlier, base_line=None, p0p1=None):
    """Estimate vanishing point using Ransac.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.

    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.

    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)
    if base_line is None:
        base_line = edgelet_lines(base_edgelet)[0]

    num_pts = strengths.size

    best_model = None
    best_votes = 0

    for i in range(num_pts):
        current_model = np.cross(base_line, lines[i])

        if np.sum(current_model**2) < 1:
            # reject degenerate candidates
            continue
        if p0p1 is not None and abs(current_model[2]) > 1e-9:
            vb0 = p0p1[1] - p0p1[0]
            vv0 = current_model[:2] / current_model[2] - p0p1[0]
            vb1 = p0p1[0] - p0p1[1]
            vv1 = current_model[:2] / current_model[2] - p0p1[1]
            vb0 /= np.linalg.norm(vb0)
            vb1 /= np.linalg.norm(vb1)
            if vb0 @ vv0 >= -50 and vb1 @ vv1 >= -50:
                continue

        current_votes = compute_votes(edgelets, current_model, threshold_inlier)
        current_votes = current_votes.sum()

        if current_votes > best_votes:
            best_model = current_model
            best_votes = current_votes

    return best_model


def ransac_vanishing_point(edgelets, threshold_inlier, num_ransac_iter):
    """Estimate vanishing point using Ransac. Modified from ransac_vanishing_point
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    num_ransac_iter: int
        Number of iterations to run ransac.
    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.
    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)
    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts//10]
    second_index_space = arg_sort[:num_pts//5]
    num_ransac_iter = min(num_ransac_iter, len(first_index_space) * len(second_index_space))

    best_model = None
    best_votes = 0

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(edgelets, current_model, threshold_inlier)
        current_votes = current_votes.sum()

        if current_votes > best_votes:
            best_model = current_model
            best_votes = current_votes

    return best_model


def result_from_two(results, infos, edgelets, remains, VP_THRES, img_shape):
    assert len(remains) == 1 and 0 not in remains
    idx = remains.pop()
    vpz = results[0]['vp']
    vpa = results[2]['vp'] if idx == 1 else results[1]['vp']
    vpz = vpz / vpz[-1]
    vpa = vpa / vpa[-1]
    cx = (img_shape[1] - 1) / 2
    cy = (img_shape[0] - 1) / 2

    lz = np.cross(vpz, [cx, cy, 1])
    A = np.array([[lz[0], lz[1]], [-lz[1], lz[0]]])
    b = np.array([-lz[2], lz[0]*vpa[1] - lz[1]*vpa[0]])
    mid = np.linalg.solve(A, b)
    base_line = np.cross(vpa, [*mid, 1])


    # import matplotlib.pyplot as plt
    # y0 = 0
    # y1 = 512
    # x0 = -(lz[1] * y0 + lz[2]) / lz[0]
    # x1 = -(lz[1] * y1 + lz[2]) / lz[0]
    # plt.plot([x0, x1], [y0, y1], 'wo-')
    # plt.plot([vpa[0], mid[0]], [vpa[1], mid[1]], 'wo-')

    if infos[idx]['type'] == 'baseline':
        # print(infos[idx]['v'])
        # print(base_line)
        vpb = np.cross(base_line, infos[idx]['v'])
    else:
        vpb = scanline_vanishing_point(edgelets, None, VP_THRES, base_line)

    inliers = detect_inliers(vpb, edgelets, VP_THRES)
    results[idx]['vp'] = vpb
    results[idx]['edgelets'] = tuple(d[inliers] for d in edgelets)
    results[idx]['type'] = 'inference from other two vps'


def detect_3vp_from_xyzinfo(img, z_info, x_info, y_info):
    VP_THRES = 5
    edgelets = compute_edgelets(img, _quant=0.7)

    infos = [z_info, x_info, y_info]
    results = [{'forced_z': 0} for _ in range(3)]
    remains = set([0, 1, 2])

    # Remove known vp
    for i in range(3):
        if infos[i]['type'] != 'vp':
            continue
        vp = infos[i]['v']
        inliers = detect_inliers(vp, edgelets, VP_THRES)
        results[i]['vp'] = vp
        results[i]['edgelets'] = tuple(d[inliers] for d in edgelets)
        if i == 0:
            results[i]['forced_z'] = 1
        results[i]['type'] = 'extract from layout'
        edgelets = tuple(d[~inliers] for d in edgelets)
        remains.remove(i)

    # if len(remains) == 1 and 0 not in remains:
    #     result_from_two(results, infos, edgelets, remains, VP_THRES, img.shape)

    # Process known baseline vp
    for i in range(3):
        if infos[i]['type'] != 'baseline':
            continue
        pt_s = np.array(infos[i]['pt_s'])
        assert pt_s.shape == (1, 2, 2)
        base_line = infos[i]['v']
        vp = scanline_vanishing_point(edgelets, None, VP_THRES, base_line, pt_s[0])
        inliers = detect_inliers(vp, edgelets, VP_THRES)
        results[i]['vp'] = vp
        results[i]['edgelets'] = tuple(d[inliers] for d in edgelets)
        if i == 0:
            results[i]['forced_z'] = 1
        results[i]['type'] = 'inference from given line'
        edgelets = tuple(d[~inliers] for d in edgelets)
        remains.remove(i)
        # if len(remains) == 1 and 0 not in remains:
        #     result_from_two(results, infos, edgelets, remains, VP_THRES, img.shape)
        #     break

    # Process unknown vp
    for i in range(3):
        if infos[i]['type'] != 'nothing':
            continue
        vp1 = ransac_vanishing_point(edgelets, VP_THRES, 2000)
        vp2 = reestimate_model(vp1, edgelets, VP_THRES)
        score1 = compute_votes(edgelets, vp1, VP_THRES).sum()
        score2 = compute_votes(edgelets, vp2, VP_THRES).sum()
        vp = vp1 if score1 > score2 else vp2
        inliers = detect_inliers(vp, edgelets, VP_THRES)
        results[i]['vp'] = vp
        results[i]['edgelets'] = tuple(d[inliers] for d in edgelets)
        results[i]['type'] = 'ransac'
        edgelets = tuple(d[~inliers] for d in edgelets)
        remains.remove(i)
        # if len(remains) == 1 and 0 not in remains:
        #     result_from_two(results, infos, edgelets, remains, VP_THRES, img.shape)
        #     break

    # Assign three vp
    results = sorted(results,
                     key=lambda v: (
                                    -v['forced_z'],
                                    np.abs(np.arctan2(v['vp'][1], v['vp'][0]) % np.pi - np.pi / 2).mean(),
                                    -abs(v['vp'][1]) / max(1e-6, abs(v['vp'][2])),
                                ))

    z_res = results[0]
    x_res = results[1]
    y_res = results[2]

    return z_res, x_res, y_res, edgelets


def compute_theta(model, edgelets):
    locations, directions, strengths = edgelets
    le = np.cross(np.hstack([locations, np.ones(len(locations))[:, None]]), model)
    est_directions = np.concatenate([-le[:, [1]], le[:, [0]]], 1)

    ve = est_directions / np.linalg.norm(est_directions, axis=1, keepdims=True)
    vd = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    theta = np.abs(np.arccos(np.clip((ve * vd).sum(axis=1), -1, 1)))
    theta = np.minimum(theta, np.pi - theta)

    return theta


def refine_model(base_model, edgelets, refine_iter=100):
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    def compute_score(model, edgelets):
        score1 = 1 - compute_theta(base_model, edgelets) / (np.pi / 2)
        # score2 = strengths / strengths.mean()
        # return (score1 * score2).sum()
        return score1.sum()

    # Random search
    best_model = base_model
    best_score = compute_score(best_model, edgelets)
    for it in range(refine_iter):
        l1 = lines[np.random.randint(len(lines))]
        l2 = lines[np.random.randint(len(lines))]
        current_model = np.cross(l1, l2)
        if np.sum(current_model**2) < 1:
            # reject degenerate candidates
            continue
        current_score = compute_score(current_model, edgelets)
        if current_score > best_score:
            best_model = current_model
            best_score = current_score

    if np.abs(best_model - base_model).sum() > 1e-9:
        print(base_model, best_model)

    return best_model


def reestimate_model(model, edgelets, threshold_reestimate):
    """Reestimate vanishing point using inliers and least squares.

    All the edgelets which are within a threshold are used to reestimate model

    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
        All edgelets from which inliers will be computed.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.

    Returns
    -------
    restimated_model: ndarry of shape (3,)
        Reestimated model for vanishing point in homogenous coordinates.
    """
    locations, directions, strengths = edgelets

    weight = compute_votes(edgelets, model, threshold_reestimate)
    inliers = weight > 0
    weight = weight[inliers]
    weight = weight / weight.sum()
    locations = locations[inliers]
    directions = directions[inliers]
    strengths = strengths[inliers]

    lines = edgelet_lines((locations, directions, strengths))
    lines = lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)

    A = lines[:, :2]
    b = -lines[:, 2]
    Aw = A * np.sqrt(weight[:, None])
    bw = b * np.sqrt(weight)
    est_model = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    pt = np.concatenate((est_model, [1.]))

    return pt


def detect_inliers(model, edgelets, threshold_inlier=1):
    return compute_votes(edgelets, model, threshold_inlier) > 0


def remove_inliers(model, edgelets, threshold_inlier=1):
    """Remove all inlier edglets of a given model.

    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.

    Returns
    -------
    edgelets_new: tuple of ndarrays
        All Edgelets except those which are inliers to model.
    """
    inliers = detect_inliers(model, edgelets, threshold_inlier)
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets


def info_from_aligned_layout(line_s, pt_s):
    if len(line_s) > 1:
        vp = []
        for i in range(len(line_s)):
            for j in range(i+1, len(line_s)):
                vp.append(np.cross(line_s[i], line_s[j]))
                if abs(vp[-1][-1]) < 1e-7:
                    vp[-1][-1] = 1e-7
                vp[-1] = vp[-1] / vp[-1][-1]
        vp = np.mean(vp, 0)
        return {'type': 'vp', 'v': vp, 'pt_s': pt_s}
    elif len(line_s) == 1:
        return {'type': 'baseline', 'v': line_s[0], 'pt_s': pt_s}
    else:
        return {'type': 'nothing', 'v': None, 'pt_s': pt_s}


def vis_edgelets(image, edgelets, show=True):
    """Helper function to visualize edgelets."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    locations, directions, strengths = edgelets
    for i in range(locations.shape[0]):
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]

        plt.plot(xax, yax, 'r-')

    if show:
        plt.show()


def vis_model(image, model, show=True):
    """Helper function to visualize computed model."""
    import matplotlib.pyplot as plt
    edgelets = compute_edgelets(image)
    locations, directions, strengths = edgelets
    inliers = compute_votes(edgelets, model, 10) > 0

    edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    locations, directions, strengths = edgelets
    vis_edgelets(image, edgelets, False)
    vp = model / model[2]
    plt.plot(vp[0], vp[1], 'bo')
    for i in range(locations.shape[0]):
        xax = [locations[i, 0], vp[0]]
        yax = [locations[i, 1], vp[1]]
        plt.plot(xax, yax, 'b-.')

    if show:
        plt.show()


def detect_3vp(img, coor0, coor1, coor2, coor3, vp_thres=1):
    layout_edgelet_1 = p0p1_2_edgelets([coor0], [coor1])
    layout_edgelet_2 = p0p1_2_edgelets([coor0], [coor2])
    layout_edgelet_3 = p0p1_2_edgelets([coor0], [coor3])
    edgelets1 = compute_edgelets(img)
    vp1 = scanline_vanishing_point(edgelets1, layout_edgelet_1, vp_thres)
    edgelets2 = remove_inliers(vp1, edgelets1, vp_thres)
    vp2 = scanline_vanishing_point(edgelets2, layout_edgelet_2, vp_thres)
    edgelets3 = remove_inliers(vp2, edgelets2, vp_thres)
    vp3 = scanline_vanishing_point(edgelets3, layout_edgelet_3, vp_thres)

    return vp1, vp2, vp3


def cal_P_scale(p0, px, vpx):
    l = np.linalg.norm(p0 - px) / 100
    A = np.array([vpx * l, -px]).T
    b = -p0
    x, e, r, s = np.linalg.lstsq(A, b, rcond=None)

    return x[0], e.sum(), l


def give_P_Z_2d_solve_3d(P, Z, coor):
    A = np.stack([P[:, 0], P[:, 1], coor], -1)
    b = -Z * P[:, 2] - P[:, 3]
    x, e, r, s = np.linalg.lstsq(A, b, rcond=None)

    return x[0], x[1], e.sum()


def construct_P(p0, p1, p2, p3, vp1, vp2, vp3):
    s1, e1, l1 = cal_P_scale(p0, p1, vp1)
    s2, e2, l2 = cal_P_scale(p0, p2, vp2)
    s3, e3, l3 = cal_P_scale(p0, p3, vp3)

    P = np.array([s1 * vp1, s2 * vp2, s3 * vp3, p0]).T

    assert e1 < 1e-6
    assert e2 < 1e-6
    assert e3 < 1e-6

    return P, l1, l2, l3


if __name__ == '__main__':

    import sys
    from skimage import io

    reestimate = False
    thres = 1
    V0 = np.array([307.12325117, 267.12858095])
    V1 = np.array([0, 331.00066622])
    V2 = np.array([308.9140573, 0])
    V3 = np.array([448, 333.38840773])
    layout_edgelet_1 = p0p1_2_edgelets([V0], [V1])
    layout_edgelet_2 = p0p1_2_edgelets([V0], [V2])
    layout_edgelet_3 = p0p1_2_edgelets([V0], [V3])

    image_name = sys.argv[-1]
    image = io.imread(image_name)

    edgelets1 = compute_edgelets(image)
    vis_edgelets(image, edgelets1)
    import sys; sys.exit()

    vp1 = scanline_vanishing_point(edgelets1, layout_edgelet_1, thres)
    if reestimate:
        vp1 = reestimate_model(vp1, edgelets1, thres)
    vis_model(image, vp1)

    # Remove inlier to remove dominating direction.
    edgelets2 = remove_inliers(vp1, edgelets1, thres)

    # Find second vanishing point
    vp2 = scanline_vanishing_point(edgelets2, layout_edgelet_2, thres)
    if reestimate:
        vp2 = reestimate_model(vp2, edgelets2, thres)
    vis_model(image, vp2)

    # Remove inlier to remove dominating direction.
    edgelets3 = remove_inliers(vp1, edgelets2, thres)

    # Find second vanishing point
    vp3 = scanline_vanishing_point(edgelets3, layout_edgelet_3, thres)
    if reestimate:
        vp3 = reestimate_model(vp3, edgelets3, thres)
    vis_model(image, vp3)
