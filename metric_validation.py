import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import math
import cv2

from PIL import Image  
import scipy
from scipy.io import loadmat, savemat
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

from models.network_plain import DnCNN 

def main():
    # Argument parsing
    opt = argparse.ArgumentParser()
    opt.add_argument('--checkpoint_path', type=str, default='./sRGB-Real-Noise-Synthesis/saves/dncnn_neca_w/models/checkpoint_G3_epoch_300_step_00232500.pth', help='checkpoint path')
    opt.add_argument('--valid_GT_dir', type=str, default='./ValidationGtBlocksSrgb.mat', help='directory of images for validation GT path')
    opt.add_argument('--valid_NOISY_dir', type=str, default='./ValidationNoisyBlocksSrgb.mat', help='directory of images for validation NOISY path')
    opt.add_argument('--NeCA_type', type=str, default='W', help='set mode of NeCA (W or S)')
    opt.add_argument('--noise_level', type=int, default=25, help='Gaussian noise level for NeCA_S')
    opt = opt.parse_args()


    """
    ### Load the Denoising model
    """
    def load_checkpoint(checkpoint_path, model):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only = True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  
        return model

    def my_srgb_denoiser(x, checkpoint_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the DnCNN model
        model = DnCNN(in_nc=3, out_nc=3, nc=64, act_mode='BR').to(device)       # Modify based on your model's architecture
        model = load_checkpoint(checkpoint_path, model)

        # Convert Image -> tensor 
        x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)  # Normalize to [0, 1]

        # Perform denoising
        with torch.no_grad():
            denoised_tensor = model(x_tensor)
        
        # Convert tensor -> numpy
        denoised_array = (
            denoised_tensor.squeeze().cpu().clamp(0, 1).mul(255.0).byte().permute(1, 2, 0).numpy()
        )

        return denoised_array
    
    """
    ### Load the Validation GT and Noisy dataset
    """
    # Input Benchmark file path
    gt_data = loadmat(opt.valid_GT_dir)['ValidationGtBlocksSrgb']
    noisy_data = loadmat(opt.valid_NOISY_dir)['ValidationNoisyBlocksSrgb']

    print(f'GT shape: {gt_data.shape}, Noisy shape: {noisy_data.shape}')

    """
    ### Denoising
    """
    # --------------------------------------------
    # PSNR Calculate Function
    # --------------------------------------------
    def calculate_psnr(img1, img2, border=0):
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')

        h, w = img1.shape[:2]
        img1 = img1[border:h-border, border:w-border]
        img2 = img2[border:h-border, border:w-border]

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    # --------------------------------------------
    # SSIM Calculate Function
    # --------------------------------------------
    def rgb2ycbcr_pt(img, y_only=False):
        """Convert RGB images to YCbCr images (PyTorch version).

        It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
        https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

        Args:
            img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
            y_only (bool): Whether to only return Y channel. Default: False.

        Returns:
            (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
        """
        if y_only:
            weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
            out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
        else:
            weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
            bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
            out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

        out_img = out_img / 255.
        return out_img

    def calculate_ssim_pt(img1, img2, crop_border=0, test_y_channel=False):
        """PyTorch based SSIM"""
        assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

        if crop_border != 0:
            img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

        if test_y_channel:
            img1 = rgb2ycbcr_pt(img1, y_only=True)
            img2 = rgb2ycbcr_pt(img2, y_only=True)

        img1 = img1.to(torch.float64)
        img2 = img2.to(torch.float64)

        return _ssim_pth(img1 * 255., img2 * 255.)

    def _ssim_pth(img1, img2):
        """Calculate SSIM (structural similarity)."""
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img1.size(1), 1, 11, 11).to(img1.dtype).to(img1.device)

        mu1 = F.conv2d(img1, window, stride=1, padding=0, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_mu2

        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

        return ssim_map.mean([1, 2, 3])

    """
    PSNR & SSIM calculate
    """
    total_psnr = 0.0
    total_ssim = 0.0
    num_blocks = gt_data.shape[0] * gt_data.shape[1]

    for i in range(gt_data.shape[0]):
        for j in range(gt_data.shape[1]):
            gt_block = gt_data[i, j]
            noisy_block = noisy_data[i, j]

            # Denoising
            denoised_block = my_srgb_denoiser(noisy_block, opt.checkpoint_path)

            # PSNR
            block_psnr = calculate_psnr(gt_block, denoised_block)
            total_psnr += block_psnr

            # SSIM 
            gt_tensor = (
                torch.from_numpy(gt_block)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255.0)
            )
            denoised_tensor = (
                torch.from_numpy(denoised_block)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255.0)
            )

            block_ssim = calculate_ssim_pt(gt_tensor, denoised_tensor)
            
            ssim_value = block_ssim.item()
            total_ssim += ssim_value

            print(f'Block ({i}, {j}) - PSNR: {block_psnr:.2f}, SSIM: {ssim_value:.4f}')

    # Average PSNR and SSIM
    avg_psnr = total_psnr / num_blocks
    avg_ssim = total_ssim / num_blocks

    print(f'\nAverage PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}') 

if __name__ == '__main__':
    main()