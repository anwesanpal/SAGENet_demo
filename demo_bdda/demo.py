# Usage: python demo.py

import sys
sys.path.append('..')
import numpy as np
import cv2

import os
from tqdm import tqdm
from os.path import join

from computer_vision_utils.io_helper import read_image

def blend_map(img, map, factor, colormap=cv2.COLORMAP_JET):

    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(map, colormap), beta=(1-factor),
                            gamma=0)

    return blend

def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return 255.0*norm_s_map

if __name__ == '__main__':

    img_shape = (1280,720)

    demo_dir = 'demo_images/'

    img_name = 'demo_img.jpg'

    im = read_image(join(demo_dir, 'frames', img_name),
                channels_first=False,
                color=True,
                dtype=np.float32,
                resize_dim = img_shape)

    bdda_pred = read_image(join(demo_dir, 'preds', 'bdda_pred.jpg'),
                channels_first=False,
                color=False,
                dtype=np.float32,
                resize_dim = img_shape)

    sage_pred = read_image(join(demo_dir, 'preds', 'sage_pred.jpg'),
                channels_first=False,
                color=False,
                dtype=np.float32,
                resize_dim = img_shape)

    im = normalize_map(im)

    bdda_pred = normalize_map(bdda_pred)
    sage_pred = normalize_map(sage_pred)

    im = im.astype(np.uint8)

    bdda_pred = bdda_pred.astype(np.uint8)
    sage_pred = sage_pred.astype(np.uint8)

    heatmap_bdda = blend_map(im, bdda_pred, factor=0.5)
    heatmap_sage = blend_map(im, sage_pred, factor=0.5)

    heatmap_bdda = cv2.resize(heatmap_bdda, img_shape)
    heatmap_sage = cv2.resize(heatmap_sage, img_shape)

    cv2.imwrite('heatmap_bdda.jpg', heatmap_bdda)
    cv2.imwrite('heatmap_sage.jpg', heatmap_sage)