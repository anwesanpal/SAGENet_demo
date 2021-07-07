# Usage: CUDA_VISIBLE_DEVICES=0 python demo.py --model_dir=pretrained_models

from network import Unet
import torch
import torchvision
import torchvision.transforms as transforms
import os
from os.path import join
import numpy as np

from PIL import Image
import cv2

class Resize_metrics(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        img = img.resize((self.size, self.size), resample=Image.BILINEAR)
        return {'image': img}

class ToTensor_metrics(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img = sample['image']
        img = self.tensor(img).unsqueeze(0)
        return {'image': img}

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
                            src2=cv2.applyColorMap(map, colormap), beta=(1-factor),
                            gamma=0)

    return blend

torch.set_printoptions(profile='full')
if __name__ == '__main__':

    model_dir = "../pretrained_models/picanet"
    models = sorted(os.listdir(model_dir), key=lambda x: int(x.split('epo_')[1].split('step')[0]))

    device = torch.device("cuda")

    bdda_model = Unet().to(device)
    sage_model = Unet().to(device)

    print("Model loaded! Loading Checkpoint...")

    bdda_model_name = models[0]
    sage_model_name = models[1]

    bdda_state_dict = torch.load(os.path.join(model_dir, bdda_model_name))
    sage_state_dict = torch.load(os.path.join(model_dir, sage_model_name))

    bdda_model.load_state_dict(bdda_state_dict)
    sage_model.load_state_dict(sage_state_dict)

    print("Checkpoint loaded! Now predicting...")

    bdda_model.eval()
    sage_model.eval()

    print('==============================')

    img_shape = (1280,720)

    demo_dir = 'demo_images'

    img_name = '{}/demo_img.jpg'.format(demo_dir)

    img = Image.open(img_name)
    img = img.convert('RGB')

    sample = {'image': img}

    transform = transforms.Compose([Resize_metrics(224), ToTensor_metrics()])
    sample = transform(sample)

    img = sample['image'].to(device)

    with torch.no_grad():
        bdda_pred, _ = bdda_model(img)
        sage_pred, _ = sage_model(img)
    
    bdda_pred = bdda_pred[5].data
    sage_pred = sage_pred[5].data

    img, bdda_pred, sage_pred = img[0].cpu().numpy(), bdda_pred[0,0].cpu().numpy(), sage_pred[0,0].cpu().numpy()

    im = np.transpose(img, (1, 2, 0))

    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

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