import json
import os
import pdb
import numpy as np

from collections import defaultdict
from tqdm import tqdm


def convert_suntos3d(data_dir, label_dir, save_dir, data_mode):
    adr = os.path.join(label_dir, f'sunrgbd_{data_mode}.json')
    if data_dir.split('/')[-1] == 'SUNRGBD':
        # parent directory
        data_dir = os.path.dirname(data_dir)
    data = json.load(open(adr))
    imgs = data['images']
    annos = data['annotations']
    imgid2anno = defaultdict(list)
    id2imgid = defaultdict(list)
    for i in range(len(annos)):
        imgid2anno[annos[i]['image_id']].append(annos[i])
    for i in range(len(imgs)):
        id2imgid[i] = imgs[i]['id']

    annotations = []
    
    for i in tqdm(range(len(imgs)), desc='Converting SUNRGBD data'):
        im_name = imgs[i]['file_name'][6:]
        im_name = os.path.join(data_dir, im_name)
        # im = cv2.imread(im_name)
        # h, w, _ = im.shape
        anno = imgid2anno[id2imgid[i]]
        sample = {}
        sample['file_name'] = im_name
        sample['layout'] = []
        for j, an in enumerate(anno):
            # seg = np.array(an['segmentation']).reshape(-1, 2)
            seg = [np.array(x).reshape(-1, 2) for x in an['segmentation']]
            line = np.array(an['inter_line']).reshape(-1,
                                                        2).astype(np.int32)
            param = np.array(an['plane_param'])
            category_id = an['category_id']
            if np.all(line == 0) or j == 0:  # no line first wall or ceiling or floor
                sample['layout'].append({
                    'attr': 0,
                    'endpoints': [],
                    'polygon': [x.tolist() for x in seg],
                    'plane_param': param.tolist(),
                    'category': category_id
                })
                continue
            if len(line) == 4:  # oc line
                left = line[0:2]
                right = line[2:]
                len_l = np.sum((left[0] - left[1])**2)
                len_r = np.sum((right[0] - right[1])**2)
                if len_l < len_r:
                    endpoints = right if right[0,
                                                1] < right[1, 1] else right[[1, 0]]
                else:
                    endpoints = left if left[0,
                                                1] < left[1, 1] else left[[1, 0]]
                sample['layout'].append({
                    'attr': 1,
                    'endpoints': endpoints.tolist(),
                    'polygon': [x.tolist() for x in seg],
                    'plane_param': param.tolist(),
                    'category': category_id
                })
                continue
            line = line if line[0, 1] < line[1, 1] else line[[1, 0]]
            sample['layout'].append({
                'attr': 2,
                'endpoints': line.tolist(),
                'polygon': [x.tolist() for x in seg],
                'plane_param': param.tolist(),
                'category': category_id
            })
        annotations.append(sample)

    with open(os.path.join(save_dir, f'sunrgbd_s3d_{data_mode}.json'), 'w') as f:
        json.dump(annotations, f)