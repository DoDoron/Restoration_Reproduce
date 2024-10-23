import os
import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from train_noise_model import init_params
from noise_model import NoiseModel
from data_loader.loader import SIDDMediumDataset
from data_loader.sidd_utils import sidd_medium_filenames_tuple
from data_loader.utils import hps_loader, ResultLogger
import argparse

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--num_workers", type=int, default=5, help="number of workers used by dataloader")
parser.add_argument("--sidd_path", type=str, help='Path to SIDD Medium Raw/SRGB dataset')
parser.add_argument("--log_dir", type=str, default='logs', help='Path to where you want to store logs')
parser.add_argument("--iso", type=int, default=None, help="ISO level for dataset images")
parser.add_argument("--cam", type=str, default=None, help="CAM type for dataset images")
parser.add_argument("--train_or_test", type=str, default=None, help="")
parser.add_argument('--model', type=str, default='DnCNN_Real', help='choose a type of model')
parser.add_argument('--noise_model_path', type=str, help='path to a noise model')
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--nm_load_epoch", type=int, default=None, help="noise model epoch to be loaded")
parser.add_argument("--synthesis_base_dir", type=str, help='')
args = parser.parse_args()
    
torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

def load_checkpoint(model, checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])
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
    saved_model_file_path = os.path.join(hps.model_save_dir, saved_model_file_name)

    nm, nm_epoch = load_checkpoint(nm, saved_model_file_path)
    print('Noise model epoch is {}'.format(nm_epoch))

# SIDD dataset setting
train_dataset = SIDDMediumDataset(
    sidd_medium_path=args.sidd_path,
    patch_sampling='dncnn',  
    train_or_test=args.train_or_test,  # train_dncnn mode
    cam=args.cam,
    iso=args.iso,
    is_raw=args.is_raw,
    first_im_idx=10,
    last_im_idx=11,
    model=nm,  # NoiseModel
	temp=hps.temp if nm else None,
	device=hps.device if nm else None
)

# DataLoader 
loader_train = DataLoader(train_dataset, batch_size=hps.n_batch_train, shuffle=False, num_workers=args.num_workers, pin_memory=True)
print('Number of training samples: {}'.format(len(train_dataset.file_names_tuple)))

# Save Path for Noise synthesis image
synthesis_base_dir = args.synthesis_base_dir  

# # Sampling and save Noisy image
for i, data in enumerate(loader_train):
    noisy_img = data['noisy']

    # Set noisy image save paths based on existing paths
    _, noisy_img_path = train_dataset.file_names_tuple[i]
    noisy_dir = os.path.dirname(noisy_img_path)
    base_name = os.path.basename(noisy_img_path)

    new_noisy_dir = os.path.join(synthesis_base_dir, os.path.relpath(noisy_dir, hps.sidd_path))
    os.makedirs(new_noisy_dir, exist_ok=True)

    new_noisy_path = os.path.join(new_noisy_dir, base_name)

    # Save noisy image 
    vutils.save_image(noisy_img, new_noisy_path, normalize=True)
    print(f"Saved synthesized noisy image to {new_noisy_path}")

