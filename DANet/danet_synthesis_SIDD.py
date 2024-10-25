#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-07-10 14:38:39

'''
In this demo, we only test the model on one image of SIDD validation dataset.
The full validation dataset can be download from the following website:
    https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php
'''

import torch
from networks import UNetG, sample_generator
from skimage import img_as_float32, img_as_ubyte
from matplotlib import pyplot as plt
from utils import PadUNet
import os
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_path", default=None, type=str)
args = parser.parse_args()

# build the network
net = UNetG(3, wf=32, depth=5).cuda()

# load the pretrained model
net.load_state_dict(torch.load('./model_states/DANet.pt', map_location='cpu')['G'])
# net.load_state_dict(torch.load('./model_states/DANetPlus.pt', map_location='cpu')['G'])

def generate_single_img(img_path, net):
    # 이미지 읽기
    img = cv2.imread(img_path, -1).astype(float)
    img = img / 255.0  # 이미지를 0-1 범위로 정규화
    
    # 이미지 배열 형태 변환: (H, W, C) -> (C, H, W)
    img = img.transpose(2, 0, 1)
    
    # NumPy 배열을 Torch 텐서로 변환하고 배치 차원 추가
    inputs = torch.from_numpy(img).float().unsqueeze(0)
    inputs = inputs.cuda()  # GPU 사용
    
    # 모델 추론
    with torch.autograd.no_grad():
        padunet = PadUNet(inputs, dep_U=5)
        inputs_pad = padunet.pad()
        outputs_pad = sample_generator(net, inputs_pad)
        outputs = padunet.pad_inverse(outputs_pad)
        outputs.clamp_(0.0, 1.0)
    
    # Torch 텐서를 NumPy 배열로 변환하고 (C, H, W) -> (H, W, C)로 변환
    im_noisy_fake = img_as_ubyte(outputs.cpu().numpy()[0].transpose(1, 2, 0))
    
    return im_noisy_fake

# 이미지 생성 경로 지정
for dirpath, _, filenames in os.walk(args.file_path):
    """
    Process only if in a folder named 'clean'
    """
    if dirpath.endswith('/clean'):
        # 원본 경로에서 상위 폴더 이름을 추출
        folder_name = dirpath.split('/')[-2]
        output_folder = f'./results/{folder_name}/noisy'
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in filenames:
            if os.path.splitext(filename)[1] not in ['.png', '.jpg', '.jpeg']:
                continue
            generated = generate_single_img(os.path.join(dirpath, filename), net)
            tag_data = os.path.splitext(filename)[0]
            tag_data = tag_data.replace('GT', 'NOISY')
            fpath_output = f'{output_folder}/{tag_data}.png'
            cv2.imwrite(fpath_output, generated)
            print('generated to %s' % (fpath_output))

# plt.subplot(1,3,1)
# plt.imshow(im_gt)
# plt.title('Gt Image')
# plt.axis('off')
# plt.subplot(1,3,2)
# plt.imshow(im_noisy_real)
# plt.title('Real Noisy Image')
# plt.axis('off')
# plt.subplot(1,3,3)
# plt.imshow(im_noisy_fake)
# plt.title('Fake Noisy Image')
# plt.axis('off')
# plt.show()

