import os
import torch
import glob
import time
import numpy as np
import argparse
import utils.utils_image as util
from models.network_unet import UNet
from models.network_plain import Noise_estimator, Noise_level_predictor

def main():
    # Argument parsing
    opt = argparse.ArgumentParser()
    opt.add_argument('--cam_name', type=str, default='IP', help='camera name for training data')
    opt.add_argument('--train_img_dir', type=str, default='./train_data/', help='directory of images for noise synthesis')
    opt.add_argument('--save_dir', type=str, default='./synthesized_images/', help='directory to save synthesized images')
    opt.add_argument('--NeCA_type', type=str, default='W', help='set mode of NeCA (W or S)')
    opt.add_argument('--noise_level', type=int, default=25, help='Gaussian noise level for NeCA_S')
    opt = opt.parse_args()

    # Load pretrained networks
    print("Load pretrained networks")
    net_E = Noise_estimator(in_nc=3, out_nc=1, nc=96, nb=5, act_mode='R')
    net_G1 = Noise_level_predictor(in_nc=3, out_nc=3, nc=96)
    net_G2 = UNet(in_nc=3, out_nc=3, nc=64, act_mode='R', num_stages=4,
                  downsample_mode='strideconv', upsample_mode='upsampling', bias=True,
                  padding_mode='zeros', final_act='Tanh')

    # Load pretrained models
    print("Load pretrained models")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained models
    print("Load pretrained models")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_E = torch.load(
        os.path.join('./pretrain/', opt.cam_name, 'checkpoint_E.pth'), 
        map_location=device, 
        weights_only=True  # Ensure only weights are loaded
    )
    checkpoint_G1 = torch.load(
        os.path.join('./pretrain/', opt.cam_name, 'checkpoint_G1.pth'), 
        map_location=device, 
        weights_only=True
    )
    checkpoint_G2 = torch.load(
        os.path.join('./pretrain/', opt.cam_name, 'checkpoint_G2.pth'), 
        map_location=device, 
        weights_only=True
    )

    net_E.load_state_dict(checkpoint_E['model_state_dict'])
    net_G1.load_state_dict(checkpoint_G1['model_state_dict'])
    net_G2.load_state_dict(checkpoint_G2['model_state_dict'])

    net_E.to(device).eval()
    net_G1.to(device).eval()
    net_G2.to(device).eval()

    def get_images_path(type, cam_name, root_dir):
        if type == 'clean':
            pattern = os.path.join(root_dir, f'**/*{cam_name}*/clean/*.png')
        else:
            pattern = os.path.join(root_dir, f'**/*{cam_name}*/noisy/*.png')
        return glob.glob(pattern, recursive=True)

    clean_images = get_images_path("clean", opt.cam_name, opt.train_img_dir)
    noisy_images = get_images_path("noisy", opt.cam_name, opt.train_img_dir)
    clean_images.sort()
    noisy_images.sort()
    
    total_start_time = time.time()

    for clean_path, noisy_path in zip(clean_images, noisy_images):
        clean = util.imread_uint(clean_path, 3)
        noisy = util.imread_uint(noisy_path, 3)

        h, w, _ = clean.shape
        if h % 8 != 0 or w % 8 != 0:
            clean = clean[:h // 8 * 8, :w // 8 * 8, :]
            noisy = noisy[:h // 8 * 8, :w // 8 * 8, :]

        clean_tensor = util.uint2tensor3(clean).unsqueeze(0).to(device)
        noisy_tensor = util.uint2tensor3(noisy).unsqueeze(0).to(device)

        # 상대 경로를 구하고 저장 경로 설정
        relative_path = os.path.relpath(os.path.dirname(noisy_path), opt.train_img_dir)
        save_subdir = os.path.join(opt.save_dir, relative_path)

        # 저장할 하위 폴더가 없으면 생성
        os.makedirs(save_subdir, exist_ok=True)

        filename = os.path.basename(noisy_path)

        with torch.no_grad():
            if opt.NeCA_type == 'W':
                z = torch.randn_like(clean_tensor).to(device)
                gain_factor = net_E(noisy_tensor)
                pred_noise_level_map = net_G1(clean_tensor, gain_factor)
                sdnu_noise = pred_noise_level_map.mul(z)
                sdnc_noise = net_G2(sdnu_noise) + sdnu_noise
                fake_noisy = clean_tensor + sdnc_noise
                
                save_path = os.path.join(save_subdir, filename)
                
                util.imsave(util.tensor2uint(fake_noisy), save_path)
                # util.imsave(util.tensor2uint(fake_noisy), os.path.join(save_subdir, f'noisy_{filename}'))
                # util.imsave(util.tensor2uint(sdnc_noise + 0.5), os.path.join(save_subdir, f'sdnc_{filename}'))
                # util.imsave(util.tensor2uint(sdnu_noise + 0.5), os.path.join(save_subdir, f'sdnu_{filename}'))

            elif opt.NeCA_type == 'S':
                z = torch.randn_like(clean_tensor).to(device)
                z.mul_(opt.noise_level / 255.0)
                sinc_noise = net_G2(z) + z
                fake_noisy = clean_tensor + sinc_noise
                
                save_path = os.path.join(save_subdir, filename)
                
                util.imsave(util.tensor2uint(fake_noisy), save_path)
                # util.imsave(util.tensor2uint(fake_noisy), os.path.join(save_subdir, f'noisy_{filename}'))
                # util.imsave(util.tensor2uint(sinc_noise + 0.5), os.path.join(save_subdir, f'sinc_{filename}'))
            else:
                raise NotImplementedError(f'NeCA_type [{opt.NeCA_type}] is not implemented')

    total_duration = time.time() - total_start_time
    print(f'Total generation finished in {total_duration:.2f} seconds')

if __name__ == '__main__':
    main()
