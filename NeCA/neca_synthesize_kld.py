import os
import torch
import glob
import time
import csv
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
    opt.add_argument('--L', type=int, default=50, help='random count of synthetic fake noisy images for AKLD')
    opt = opt.parse_args()

    def get_image_paths(directory, type="PolyU", cam_name=None):
        # 데이터 안에서 cam 구분을 하지 않아도 되는 dataset
        if cam_name == None:
            # PolyU
            if type == "PolyU":
                clean_images, noisy_images = [], []
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if 'mean' in filename:
                        clean_images.append(filepath)
                    elif 'real' in filename.lower():
                        noisy_images.append(filepath)
                return clean_images, noisy_images
            # SIDD+
            elif type == "siddplus":
                def glob_pattern(img_type):
                    subfolder = "gt" if img_type == "gt" else "noisy"
                    return os.path.join(directory, f'**/{subfolder}/*.png')
                return glob.glob(glob_pattern("gt"), recursive=True), glob.glob(glob_pattern("noisy"), recursive=True)
            # Nam
            elif type == "nam":
                print("get_images_path")
                def glob_pattern(img_type):
                    subfolder = "hq" if img_type == "hq" else "lq"
                    return os.path.join(directory, f'**/{subfolder}/*.png')
                return glob.glob(glob_pattern("hq"), recursive=True), glob.glob(glob_pattern("lq"), recursive=True)
        
        # 데이터 안에서 cam 구분을 해야 하는 dataset
        # SIDD sRGB Medium
        else:
            def glob_pattern(img_type):
                subfolder = "clean" if img_type == "clean" else "noisy"
                return os.path.join(directory, f'**/*{cam_name}*/{subfolder}/*.png')

            return glob.glob(glob_pattern("clean"), recursive=True), glob.glob(glob_pattern("noisy"), recursive=True)

    """
    Clean & Noisy images load
    """
    if 'PolyU' in opt.train_img_dir:
        data_type = "PolyU"
    if 'siddplus' in opt.train_img_dir or 'SIDD_validation' in opt.train_img_dir:
        data_type = "siddplus"
    if 'nam_512' in opt.train_img_dir:
        print("nam_512 enter")
        data_type = "nam"
    clean_images, noisy_images = get_image_paths(opt.train_img_dir, data_type, opt.cam_name if 'SIDD_srgb_medium' in opt.train_img_dir else None)
    clean_images.sort()
    noisy_images.sort()

    """
    Load pretrained networks
    """
    print("Load pretrained networks")
    net_E = Noise_estimator(in_nc=3, out_nc=1, nc=96, nb=5, act_mode='R')
    net_G1 = Noise_level_predictor(in_nc=3, out_nc=3, nc=96)
    net_G2 = UNet(in_nc=3, out_nc=3, nc=64, act_mode='R', num_stages=4,
                  downsample_mode='strideconv', upsample_mode='upsampling', bias=True,
                  padding_mode='zeros', final_act='Tanh')
    
    """
    Load pretrained models
    """
    print("Load pretrained models")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_E = torch.load(
        os.path.join('./pretrain/', opt.cam_name, 'checkpoint_E.pth'), 
        map_location=device 
        # weights_only=True  # Ensure only weights are loaded
    )
    checkpoint_G1 = torch.load(
        os.path.join('./pretrain/', opt.cam_name, 'checkpoint_G1.pth'), 
        map_location=device 
        # weights_only=True
    )
    checkpoint_G2 = torch.load(
        os.path.join('./pretrain/', opt.cam_name, 'checkpoint_G2.pth'), 
        map_location=device 
        # weights_only=True
    )

    net_E.load_state_dict(checkpoint_E['model_state_dict'])
    net_G1.load_state_dict(checkpoint_G1['model_state_dict'])
    net_G2.load_state_dict(checkpoint_G2['model_state_dict'])

    net_E.to(device).eval()
    net_G1.to(device).eval()
    net_G2.to(device).eval()

    """
    KLD & AKLD 
    """
    total_kl = 0
    total_akld = 0
    test_num = 0
    total_start_time = time.time()

    # Prepare CSV file for writing
    if not os.path.exists(opt.save_dir): # no resume
        os.makedirs(opt.save_dir)
    csv_file_path = os.path.join(opt.save_dir, f"results_{opt.cam_name}.csv")       # cam name 별로 결과 저장

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["Fake Noisy", "KLD", "AKLD"])

        for clean_path, noisy_path in zip(clean_images, noisy_images):
            test_num += 1
            clean = util.imread_uint(clean_path, 3)
            noisy = util.imread_uint(noisy_path, 3)
            h, w, _ = clean.shape
            if h % 8 != 0 or w % 8 != 0:
                clean = clean[:h // 8 * 8, :w // 8 * 8, :]
                noisy = noisy[:h // 8 * 8, :w // 8 * 8, :]

            clean_tensor = util.uint2tensor3(clean).unsqueeze(0).to(device)
            noisy_tensor = util.uint2tensor3(noisy).unsqueeze(0).to(device)
            noise_tensor = noisy_tensor - clean_tensor

            # Get relative path and save directory
            relative_path = os.path.relpath(os.path.dirname(noisy_path), opt.train_img_dir)
            save_subdir = os.path.join(opt.save_dir, opt.cam_name, relative_path)
        
            if not os.path.exists(opt.save_dir): # no resume
                os.makedirs(opt.save_dir, exist_ok=True)
            if not os.path.exists(save_subdir): # no resume
                os.makedirs(save_subdir, exist_ok=True)

            filename = os.path.basename(noisy_path)

            # Initialize variables for AKLD and KLD calculations
            akld = 0
            kld = 0
            sigma_real = util.estimate_sigma_gauss(noisy_tensor, clean_tensor)
            
            with torch.no_grad():
                for i in range(opt.L):
                    if opt.NeCA_type == 'W':
                        z = torch.randn_like(clean_tensor).to(device)
                        gain_factor = net_E(noisy_tensor)
                        pred_noise_level_map = net_G1(clean_tensor, gain_factor)
                        sdnu_noise = pred_noise_level_map * z
                        sdnc_noise = net_G2(sdnu_noise) + sdnu_noise
                        fake_noisy = clean_tensor + sdnc_noise

                    elif opt.NeCA_type == 'S':
                        z = torch.randn_like(clean_tensor).to(device)
                        z.mul_(opt.noise_level / 255.0)
                        sinc_noise = net_G2(z) + z
                        fake_noisy = clean_tensor + sinc_noise

                    else:
                        raise NotImplementedError(f'NeCA_type [{opt.NeCA_type}] is not implemented')

                    # Calculate AKLD
                    clamp_fake_noisy = torch.clip(fake_noisy, 0, 1)
                    sigma_fake = util.estimate_sigma_gauss(clamp_fake_noisy, clean_tensor)
                    kl_dis = util.kl_gauss_zero_center(sigma_fake, sigma_real)
                    akld += kl_dis

                    # Calculate KLD
                    if i == 0:
                        # fake_noise = fake_noisy - clean_tensor
                        # print("###################################")
                        # print("Noisy tensor min:", fake_noisy.min().item())
                        # print("Noisy tensor max:", fake_noisy.max().item())
                        # print()
                        # print("Noise tensor min:", fake_noise.min().item())
                        # print("Noise tensor max:", fake_noise.max().item())
                        
                        quantized_noise = util.noise_quantization(fake_noisy, clean_tensor)     # quantized noise : [ noisy -> clip(0,1) -> round(*255)/255 ] - clean
                        
                        # p_data = util.tensor2data(noise_tensor)
                        # q_data = util.tensor2data(quantized_noise)
                        # print("p_data range:", p_data.min().item(), p_data.max().item())
                        # print("q_data range:", q_data.min().item(), q_data.max().item())

                        kld = util.cal_kld(util.tensor2data(noise_tensor), util.tensor2data(quantized_noise))
                        util.imsave(util.tensor2uint(fake_noisy), os.path.join(save_subdir, f'{filename}'))

            # Average AKLD and update totals
            akld /= opt.L
            total_akld += akld
            total_kl += kld

            # Write row to CSV file
            csv_writer.writerow([filename, f"{kld:.4f}", f"{akld:.4f}"])              

    total_kl /= test_num
    total_akld /= test_num
    print("KLD : {:.4e} / AKLD : {:.4e}".format(total_kl, total_akld))

    total_duration = time.time() - total_start_time
    print(f'Total generation finished in {total_duration:.2f} seconds')

if __name__ == '__main__':
    main()
