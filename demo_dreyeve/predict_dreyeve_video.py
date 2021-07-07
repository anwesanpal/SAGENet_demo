# Usage: THEANO_FLAGS=device=cuda python predict_dreyeve_video.py --vid_in=../videos/demo_video.mp4 --gt_type='sage'

import numpy as np
import cv2

import argparse

import os
from tqdm import tqdm
from os.path import join

import sys
sys.path.append('..')
from models import SaliencyBranch
from computer_vision_utils.io_helper import read_image, normalize
from computer_vision_utils.tensor_manipulation import resize_tensor

import skvideo.io

resize_dim_in = (448,448)
resize_dim_disp = (540,960)

def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return 255.0*norm_s_map

def blend_map(img, map, factor, colormap=cv2.COLORMAP_JET):
    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(255-map, colormap), beta=(1-factor),
                            gamma=0)

    return blend

def load_dreyeve_sample(frames_list, sample, mean_dreyeve_image, frames_per_seq=16, h=448, w=448, ):
    h_c = h_s = h // 4
    w_c = w_s = h // 4

    I_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    I_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    I_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    for fr in range(frames_per_seq):
        offset = sample - frames_per_seq + 1 + fr   # tricky

        x_in = cv2.resize(frames_list[offset], dsize=resize_dim_in[::-1], interpolation=cv2.INTER_LINEAR)
        x_in = np.transpose(x_in, (2, 0, 1)).astype(np.float32)
        x_in -= mean_dreyeve_image

        I_s[0, :, fr, :, :] = resize_tensor(x_in, new_size=(h_s, w_s))

        x_disp = cv2.resize(frames_list[offset], dsize=resize_dim_disp[::-1], interpolation=cv2.INTER_LINEAR)
        
    I_ff[0, :, 0, :, :] = x_in

    return [I_ff, I_s, I_c], x_disp

if __name__ == '__main__':

    frames_per_seq, h, w = 16, 448, 448
    verbose = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_in", type=str)
    parser.add_argument('--gt_type', type=str)
    args = parser.parse_args()

    assert args.vid_in is not None, 'Please provide a correct video path'

    print("Reading video...")
    frames, pred_list = [],[]
    vidcap = cv2.VideoCapture(args.vid_in)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        frames.append(image)
        success,image = vidcap.read()

    print('Finished reading video!')

    print('Now starting prediction...')

    demo_dir = 'demo_images'

    # load mean dreyeve image
    mean_dreyeve_image = read_image(join(demo_dir, 'dreyeve_mean_frame.png'),
                                    channels_first=True, resize_dim=(h, w))

    # get the models

    image_branch = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='image')
    image_branch.compile(optimizer='adam', loss='kld')
    if(args.gt_type == 'sage'):
        image_branch.load_weights('../pretrained_models/dreyeve/sage_image_branch.h5')  # load weights
    else:
        image_branch.load_weights('../pretrained_models/dreyeve/bdda_image_branch.h5')  # load weights


    for sample in tqdm(range(15, len(frames))):
        from time import time
        t = time()
        X, im = load_dreyeve_sample(frames_list=frames, sample=sample, mean_dreyeve_image=mean_dreyeve_image,
                                frames_per_seq=frames_per_seq, h=h, w=w)

        Y_image = image_branch.predict(X[:3])[0]  # predict on image
        my_pred = np.squeeze(np.squeeze(Y_image, axis=0),axis=0)
        my_pred = cv2.resize(my_pred, dsize=resize_dim_disp[::-1])

        im = normalize_map(im)
        my_pred = normalize_map(my_pred)
        im = im.astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        my_pred = my_pred.astype(np.uint8)
        my_pred = blend_map(im, my_pred, factor=0.5)
        pred_list.append(my_pred)

print('Prediction complete! Now writing...')
if(args.gt_type == 'sage'):
    writer = skvideo.io.FFmpegWriter("sage_demo_video.mp4", outputdict={
      '-vcodec': 'libx264',
      '-pix_fmt': 'yuv420p',
      '-r': str(fps),
})
else:
    writer = skvideo.io.FFmpegWriter("bdda_demo_video.mp4", outputdict={
      '-vcodec': 'libx264',
      '-pix_fmt': 'yuv420p',
      '-r': str(fps),
})
 
for i in range(len(pred_list)):
    writer.writeFrame(pred_list[i])
writer.close()