# non sidd synthesis
import os
import glob
import time
import csv
import cv2
import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from train_noise_model import init_params
from noise_model import NoiseModel
from data_loader.iterable_loader import IterableDataset
from data_loader.utils import hps_loader, ResultLogger
from torchvision.transforms.functional import to_pil_image
import argparse
from data_loader.sidd_utils import calc_kldiv_mb
import utils.utils_image as util

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers used by dataloader")
parser.add_argument("--data_path", type=str, help='Path to SIDD Medium Raw/SRGB dataset')
parser.add_argument("--log_dir", type=str, default='logs', help='Path to where you want to store logs')
parser.add_argument("--iso", type=int, default=None, help="ISO level for dataset images")
parser.add_argument("--cam", type=str, default=None, help="CAM type for dataset images")
parser.add_argument("--train_or_test", type=str, default=None, help="")
parser.add_argument('--model', type=str, default='DnCNN_Real', help='choose a type of model')
parser.add_argument('--model_save_dir', type=str)
parser.add_argument('--noise_model_path', type=str, help='path to a noise model')
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--nm_load_epoch", type=int, default=None, help="noise model epoch to be loaded")
parser.add_argument("--synthesis_base_dir", type=str, help='')
args = parser.parse_args()
    
torch.random.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_checkpoint(model, checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # grad_scaler.load_state_dict(checkpoint['grad_scaler'])
    return model, checkpoint['epoch_num']

# NoiseModel and checkpoint load
args.is_raw = False
args.n_channels = 4 if args.is_raw else 3
data_range = 1. if args.is_raw else 255.
x_shape = [4, 32, 32] if args.is_raw else [3, 32, 32]

nm = None # Noise model
if args.model.__contains__('DnCNN_NM'):
    nm_path = os.path.abspath(os.path.join('experiments', 'sidd', args.noise_model_path))
    print(nm_path)
    hps = hps_loader(os.path.join(nm_path, 'hps.txt'))
    hps.param_inits = init_params()
    nm = NoiseModel(x_shape, hps.arch, hps.flow_permutation, hps.param_inits, hps.lu_decomp, hps.device, False)
    nm.to(hps.device)

    logdir = nm_path + '/saved_models'
    models = sorted(os.listdir(hps.model_save_dir))
    if args.nm_load_epoch:
        last_epoch = args.nm_load_epoch
    else:
        last_epoch = str(max([int(i.split("_")[1]) for i in models[1:]]))
    saved_model_file_name = 'epoch_{}_noise_model_net.pth'.format(last_epoch)
    saved_model_file_path = os.path.join(args.model_save_dir, saved_model_file_name)

    nm, nm_epoch = load_checkpoint(nm, saved_model_file_path)
    print('Noise model epoch is {}'.format(nm_epoch))

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
       
        # SIDD validation with ISO filtering
        elif type == "siddvalid":
            iso_vals = [100, 400, 800, 1600, 3200]
            def glob_pattern(img_type):
                subfolder = "gt" if img_type == "gt" else "noisy"
                return os.path.join(directory, f'**/{subfolder}/*.png')

            clean_images, noisy_images = [], []
            for filepath in glob.glob(glob_pattern("noisy"), recursive=True):
                filename = os.path.basename(filepath)
                parts = filename.split("_")
                # ISO 값이 3번째에 있다고 가정하고 텍스트로 비교
                iso_value_str = int(parts[3])
                if iso_value_str in [iso for iso in iso_vals]:
                    noisy_images.append(filepath)
            
            for filepath in glob.glob(glob_pattern("gt"), recursive=True):
                filename = os.path.basename(filepath)
                parts = filename.split("_")
                # ISO 값이 3번째에 있다고 가정하고 텍스트로 비교
                iso_value_str = int(parts[3])
                if iso_value_str in [iso for iso in iso_vals]:
                    clean_images.append(filepath)
            
            return clean_images, noisy_images
        # Nam
        elif type == "nam":
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
if 'PolyU' in args.data_path:
    data_type = "PolyU"
if 'siddplus' in args.data_path:
    data_type = "siddplus"
if 'SIDD_validation' in args.data_path: data_type = "siddvalid"
if 'nam_512' in args.data_path:
    data_type = "nam"

clean_images, noisy_images = get_image_paths(args.data_path, data_type, None)
clean_images.sort()
print(len(clean_images))
noisy_images.sort()
print(len(noisy_images))

# Create a dictionary to match the base names
if 'PolyU' in args.data_path:
    clean_dict = {os.path.basename(f).replace('_mean.JPG', ''): f for f in clean_images}
    noisy_dict = {os.path.basename(f).replace('_real.JPG', ''): f for f in noisy_images}
if 'siddplus' in args.data_path or 'SIDD_validation' in args.data_path:
    clean_dict = {os.path.basename(f).replace('.png', ''): f for f in clean_images}
    noisy_dict = {os.path.basename(f).replace('.png', ''): f for f in noisy_images}

# Pair mean and real images in tuples
fns = []
for base_name in clean_dict:
    if base_name in noisy_dict:
        fns.append((noisy_dict[base_name], clean_dict[base_name]))
cnt_inst = len(fns)
test_dataset = IterableDataset(
    data_path=args.data_path,
    train_or_test='test',
    cam=hps.camera,
    iso=hps.iso,
    patch_size=(32, 32),
    is_raw=False,
    num_patches_per_image = 64,
    file_names_tuple = fns,
    cnt_inst = cnt_inst
)

# DataLoader 
test_dataloader = DataLoader(test_dataset, batch_size=hps.n_batch_test, shuffle=False, num_workers=hps.num_workers, pin_memory=True)
hps.n_ts_inst = test_dataset.cnt_inst

# Save Path for Noise synthesis image
synthesis_base_dir = args.synthesis_base_dir  

def save_image(tensor, save_path):
    img = to_pil_image(tensor[0].cpu())  # 배치 차원 제거
    img.save(save_path)

test_loss_best = np.inf

sample_loss = []
sample_sdz = []
sample_time = 0
n_models = 2
kldiv = np.zeros(n_models)
sample_loss_mean = 0
"""
modify fix iso and cam
"""
is_fix = False  # to fix the camera and ISO
iso_vals = [100, 400, 800, 1600, 3200]
iso_fix = [100]
cam_fix = [['IP', 'GP', 'S6', 'N6', 'G4'].index('S6')]
nlf_s6 = [[0.000479, 0.000002], [0.001774, 0.000002], [0.003696, 0.000002], [0.008211, 0.000002],
        [0.019930, 0.000002]]

sample_curr_time = time.time()
nm.eval()

total_kl = 0
total_akld = 0
count = 0
# Prepare CSV file for writing
if not os.path.exists('./results'): # no resume
    os.makedirs('./results')
csv_file_path = os.path.join('./results', f"results.csv")
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header
    csv_writer.writerow(["Fake Noisy", "KLD", "AKLD"])
        
    for n_batch, batch_images in enumerate(test_dataloader):
        step = (nm_epoch - 1) * (test_dataset.__len__()/hps.n_batch_test) + n_batch
        count +=1
        clean_images = batch_images['clean'].to(device)
        clean_tensor = clean_images.float().div(255.)
        noisy_images = batch_images['noisy'].to(device)
        noisy_tensor = noisy_images.float().div(255.)
        noise_images = batch_images['noise'].to(device)
        noise_tensor = noisy_tensor - clean_tensor

        kwargs = {
            'clean': clean_images,
            'eps_std': torch.tensor(1.0, device=device),
            # 'writer': writer,
            # 'step': step
        }

        if is_fix:
            kwargs.update({
                'iso': iso_fix[0].to(device),
                'cam': cam_fix[0].to(device),
                'nlf0': [nlf_s6[iso_vals.index(iso_fix[0])][0]],
                'nlf1': [nlf_s6[iso_vals.index(iso_fix[0])][0]]
            })
        else:
            print(f"before sampling : {count}")
            kwargs.update({
                'iso': batch_images['iso'].to(hps.device),
                'cam': batch_images['cam'].to(hps.device)
            })

            if 'nlf0' in batch_images.keys():
                kwargs.update({
                    'nlf0': batch_images['nlf0'].to(hps.device),
                    'nlf1': batch_images['nlf1'].to(hps.device)
                })

        # image_patch_path = batch_images['image_path']
        # print()
        # print(image_patch_path)
        # relative_path = os.path.relpath(image_patch_path, args.data_path)

        # # 새로운 디렉토리 경로 생성
        # new_noisy_dir = os.path.join(synthesis_base_dir, os.path.dirname(relative_path))
        # os.makedirs(new_noisy_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        # base_name = os.path.basename(image_patch_path)
        # new_noisy_path = os.path.join(new_noisy_dir, base_name)
        # print("directory name")
        # print(new_noisy_dir)
        # print(new_noisy_path)
                
        akld = 0
        kld = 0
        sigma_real = util.estimate_sigma_gauss(noisy_tensor, clean_tensor)

        with torch.no_grad():
            for i in range(50):
                x_sample_val = nm.sample(**kwargs)
                kwargs.update({'x': x_sample_val})

                fake_noisy = x_sample_val.float().div(255.)
                clamp_fake_noisy = torch.clip(fake_noisy, 0, 1)
                sigma_fake = util.estimate_sigma_gauss(clamp_fake_noisy, clean_tensor)
                kl_dis = util.kl_gauss_zero_center(sigma_fake, sigma_real)
                akld += kl_dis

                # Calculate KLD
                if i == 0:
                    quantized_noise = util.noise_quantization(fake_noisy, clean_tensor)     # quantized noise : [ noisy -> clip(0,1) -> round(*255)/255 ] - clean
                    kld = util.cal_kld(util.tensor2data(noise_tensor), util.tensor2data(quantized_noise))
                    util.imsave(util.tensor2uint(fake_noisy), os.path.join(synthesis_base_dir, f'{count}.png'))

        # Average AKLD and update totals
        akld /= 50
        total_akld += akld
        total_kl += kld
        # Write row to CSV file
        csv_writer.writerow([count, f"{kld:.4f}", f"{akld:.4f}"])   

total_kl /= count
total_akld /= count
print(f"KLD : {total_kl} / AKLD : {total_akld}")

    # # Save noisy image 
    # vutils.save_image(x_sample_val, new_noisy_path, normalize=True)
    # print(f"Saved synthesized noisy image to {new_noisy_path}")
