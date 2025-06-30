import pathlib
import numpy as np
import cv2
import pdb

from collections import namedtuple
from typing import Sequence
from scipy.io import loadmat
from pathlib import Path
from PIL import Image

Scene = namedtuple('Scene', ['filename', 'scene_type', 'layout_type', 'keypoints', 'shape'])

LSUN_keypoint_struc = {
    0: {
        '# corners': 8,
        'ceiling': [2, 1, 7, 8],
        'floor': [4, 3, 5, 6],
        'ceiling-wall': [1, 7],
        'floor-wall': [3, 5],
    },
    1: {
        '# corners': 6,
        'ceiling': [],
        'floor': [3, 1, 4, 6],
        'ceiling-wall': [2, 5],
        'floor-wall': [1, 4],
    },
    2: {
        '# corners': 6,
        'ceiling': [2, 1, 4, 5],
        'floor': [],
        'ceiling-wall': [1, 4],
        'floor-wall': [3, 6],
    },
    3: {
        '# corners': 4,
        'ceiling': [2, 1, 4],
        'floor': [],
        'ceiling-wall': [1],
        'floor-wall': [3],
    },
    4: {
        '# corners': 4,
        'ceiling': [],
        'floor': [2, 1, 4],
        'ceiling-wall': [3],
        'floor-wall': [1],
    },
    5: {
        '# corners': 6,
        'ceiling': [2, 1, 3],
        'floor': [5, 4, 6],
        'ceiling-wall': [1],
        'floor-wall': [4],
    },
    6: {
        '# corners': 4,
        'ceiling': [1, 2],
        'floor': [3, 4],
        'ceiling-wall': [],
        'floor-wall': [],
    },
    7: {
        '# corners': 4,
        'ceiling': [],
        'floor': [],
        'ceiling-wall': [1, 3],
        'floor-wall': [2, 4],
    },
    8: {
        '# corners': 2,
        'ceiling': [1, 2],
        'floor': [],
        'ceiling-wall': [],
        'floor-wall': [],
    },
    9: {
        '# corners': 2,
        'ceiling': [],
        'floor': [1, 2],
        'ceiling-wall': [],
        'floor-wall': [],
    },
    10: {
        '# corners': 2,
        'ceiling': [],
        'floor': [],
        'ceiling-wall': [1],
        'floor-wall': [2],
    },
}

def read_image(image_path, shape):
    img = np.array(Image.open(image_path))
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_AREA)

    return img

def load_lsun_mat(filepath: pathlib.Path) -> Sequence[Scene]:
    data = loadmat(filepath)

    return [
        Scene(*(m.squeeze() for m in metadata))
        for metadata in data[Path(filepath).stem].squeeze()
    ]

def resize_keypoints(c_c, c_f, c_cw, c_fw, ori_shape, req_shape):
    c_c[:, 0] = c_c[:, 0] / ori_shape[1] * req_shape[1]
    c_c[:, 1] = c_c[:, 1] / ori_shape[0] * req_shape[0]
    c_f[:, 0] = c_f[:, 0] / ori_shape[1] * req_shape[1]
    c_f[:, 1] = c_f[:, 1] / ori_shape[0] * req_shape[0]
    c_cw[:, 0] = c_cw[:, 0] / ori_shape[1] * req_shape[1]
    c_cw[:, 1] = c_cw[:, 1] / ori_shape[0] * req_shape[0]
    c_fw[:, 0] = c_fw[:, 0] / ori_shape[1] * req_shape[1]
    c_fw[:, 1] = c_fw[:, 1] / ori_shape[0] * req_shape[0]
    return c_c, c_f, c_cw, c_fw