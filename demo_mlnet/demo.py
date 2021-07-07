# Usage: THEANO_FLAGS=device=cuda python demo.py

import sys
sys.path.append('..')
import numpy as np
import cv2

import os
from os.path import join

from model import ml_net_model
from computer_vision_utils.io_helper import normalize, read_image

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

def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims.transpose((0, 3, 1, 2))

    return ims

def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img / np.max(img) * 255

def load_dreyeve_sample(sequence_dir, sample, shape_r, shape_c):
    filename = join(sequence_dir, 'frames', '{}.jpg'.format(sample))
    X = preprocess_images([filename], shape_r, shape_c)

    return X

if __name__ == '__main__':

    shape_r = 480
    shape_c = 640

    # get the model
    model_bdda = ml_net_model(img_rows=shape_r, img_cols=shape_c)
    model_bdda.compile(optimizer='adam', loss='kld')

    model_sage = ml_net_model(img_rows=shape_r, img_cols=shape_c)
    model_sage.compile(optimizer='adam', loss='kld')

    model_path_bdda = "../pretrained_models/mlnet/weights.mlnet.bdda.pkl"
    model_path_sage = "../pretrained_models/mlnet/weights.mlnet.sage.pkl"

    model_bdda.load_weights(model_path_bdda)
    model_sage.load_weights(model_path_sage)

    demo_dir = 'demo_images/'
    X = load_dreyeve_sample(sequence_dir=demo_dir, sample=16, shape_c=shape_c, shape_r=shape_r)

    # predict sample
    P_bdda = model_bdda.predict(X)
    P_bdda = np.squeeze(P_bdda)

    P_bdda = postprocess_predictions(P_bdda, shape_r, shape_c)

    P_sage = model_sage.predict(X)
    P_sage = np.squeeze(P_sage)

    P_sage = postprocess_predictions(P_sage, shape_r, shape_c)

    im = read_image(join(demo_dir, 'demo_img.jpg'),
                                    channels_first=True)
    h, w = im.shape[1], im.shape[2]

    bdda_pred = P_bdda
    sage_pred = P_sage

    bdda_pred = cv2.resize(bdda_pred, dsize=(w, h))
    sage_pred = cv2.resize(sage_pred, dsize=(w, h))

    im = normalize_map(im)
    bdda_pred = normalize_map(bdda_pred)
    sage_pred = normalize_map(sage_pred)

    im = im.astype(np.uint8)
    bdda_pred = bdda_pred.astype(np.uint8)
    sage_pred = sage_pred.astype(np.uint8)

    im = np.transpose(im, (1, 2, 0))

    heatmap_bdda = blend_map(im, bdda_pred, factor=0.5)
    heatmap_sage = blend_map(im, sage_pred, factor=0.5)

    cv2.imwrite(join(demo_dir, 'heatmap_bdda.jpg'), heatmap_bdda)
    cv2.imwrite(join(demo_dir, 'heatmap_sage.jpg'), heatmap_sage)