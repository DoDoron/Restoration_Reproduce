import torch
import torch.nn as nn
import logging
import os
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from collections import OrderedDict
import time

from options.option_noise_synthesis import BaseOptions
from utils.utils_logger import logger_info
from data.dataset_noise_synthesis import dataset_noise_synthesis
from models.network_plain import DnCNN 
import utils.utils_network as net_util
import utils.utils_image as util


if __name__ == '__main__':
    """
    # -----------------------------------------------
    # step 1 -- initiate logs and args
    # -----------------------------------------------
    """
    # get training options
    opt, opt_message = BaseOptions().parse() 

    # initiate data folder
    cam_name = opt.cam_name
    dir_save_logs = os.path.join(opt.dir_save, 'logs/')
    dir_save_imgs = os.path.join(opt.dir_save, 'imgs/')
    dir_save_models = os.path.join(opt.dir_save, 'models/')
    dir_save_tb = os.path.join(opt.dir_save, 'tb/')

    if not os.path.exists(dir_save_logs):
        os.makedirs(dir_save_logs)
    if not os.path.exists(dir_save_imgs):
        os.makedirs(dir_save_imgs)
    if not os.path.exists(dir_save_models):
        os.makedirs(dir_save_models)
    if not os.path.exists(dir_save_tb):
        os.makedirs(dir_save_tb)

    # initiate logs
    logger_name = 'train'
    logger_info(logger_name, log_path=dir_save_logs+logger_name+'.log')
    logger = logging.getLogger(logger_name)
    logger.info(opt_message)
    
    """
    # -----------------------------------------------
    # step 2 -- prepare dataloader
    # -----------------------------------------------
    """
    
    if opt.generate_data_lists:     # data_preparation파일이 있음으로 안해도 O
        util.train_test_split(opt.train_img_dir, opt.data_prep_dir, cam_name=cam_name, ratio=0.8)
    
    train_img_lists, val_img_lists = util.get_img_list(opt.train_img_dir, cam_name=cam_name, mode='train', ratio=1)
    test_img_lists = util.get_img_list(opt.train_img_dir, cam_name=cam_name, mode='test')
    data_message = 'train: {:d}, val: {:d}, test: {:d}'.format(len(train_img_lists), len(val_img_lists), len(test_img_lists))       # train: 24800, val: 0, test: 5808
    print(data_message)
    logger.info(data_message)

    # dataset_noise_synthesis에 random crop 존재
    train_data = dataset_noise_synthesis(img_lists=train_img_lists,
                                         img_channels=opt.img_channels,
                                         not_train=opt.not_train,
                                         not_aug=opt.not_aug,
                                         patch_size=opt.patch_size)
    val_data = dataset_noise_synthesis(img_lists=val_img_lists,
                                       img_channels=opt.img_channels,
                                       not_train=True)
    test_data = dataset_noise_synthesis(img_lists=test_img_lists,
                                        img_channels=opt.img_channels,
                                        not_train=True)
    
    train_loader = DataLoader(dataset=train_data,
                              batch_size=opt.batch_size,
                              shuffle=not opt.not_train_shuffle,
                              num_workers=opt.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=False)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=opt.num_workers,
                             pin_memory=True,
                             drop_last=False) 
    
    """
    # ------------------------------------------------
    # step 3 -- prepare DnCNN and optimizer
    # ------------------------------------------------ 
    """
    # prepare DnCNN as net_G3
    net_G3 = DnCNN(in_nc=opt.G3_in_nc, 
                   out_nc=opt.G3_out_nc,
                   nc=opt.G3_nc,
                   act_mode=opt.G3_act_mode)

    # print DnCNN
    net_message = '------------ DnCNN Network params -----------\n'
    net_message += net_util.print_networks(network=net_G3, network_name='Denoiser', verbose=opt.verbose)
    net_message += '-----------------------------------------------\n'
    logger.info(net_message)

    # send network to GPU
    net_G3, gpu_message, device = net_util.model_to_device(net_G3, gpu_ids=opt.gpu_ids)
    logger.info(gpu_message)
    
    # set optimizer
    optim_G3 = optim.Adam(net_G3.parameters(), lr=opt.lr_G3, betas=(opt.beta1, opt.beta2))

    # set learning rate scheduler
    lr_scheduler_G3 = optim.lr_scheduler.MultiStepLR(optimizer=optim_G3, milestones=opt.lr_decay_2, gamma=0.1)

    # set loss function (mean squared error)
    Cri_rec = nn.MSELoss()

    # load pretrained network if exists
    if opt.use_pretrained:
        checkpoint_G3 = torch.load(os.path.join(dir_save_models, opt.checkpoint_G3_name), map_location=device)
        net_G3.load_state_dict(checkpoint_G3['model_state_dict'])
        optim_G3.load_state_dict(checkpoint_G3['optimizer_state_dict'])
        lr_scheduler_G3.load_state_dict(checkpoint_G3['scheduler_state_dict'])
        start_epoch = checkpoint_G3['epoch'] + 1
        current_step = checkpoint_G3['step']  # Load current step from checkpoint
    else:
        net_util.init_weights(net_G3, init_type=opt.init_type)
        start_epoch = 1
        current_step = 0

    """
    # ------------------------------------------------
    # step 4 -- training DnCNN with all data
    # ------------------------------------------------
    """
    # Record the total training start time
    total_start_time = time.time()

    for epoch in range(start_epoch, opt.epochs+1):
        epoch_start_time = time.time()  # Record the start time of the epoch
        
        # Train phase
        net_G3.train()
        train_loss = 0.0    
        for i, data in enumerate(train_loader):
            current_step += 1
            clean = data['clean'].to(device)
            noisy = data['noisy'].to(device)

            # optimize denoiser
            optim_G3.zero_grad()
            rec_clean = net_G3(noisy)
            rec_loss = Cri_rec(rec_clean, clean)
            rec_loss.backward()
            optim_G3.step()
            train_loss += rec_loss.item()

        train_loss /= len(train_loader)

        # Test phase
        if epoch % opt.test_epoch == 0:
            net_G3.eval()
            test_message = OrderedDict()
            test_psnr = 0
            test_num = 0
            
            for data in test_loader:
                test_num += 1
                clean = data['clean'].to(device)
                noisy = data['noisy'].to(device)
                with torch.no_grad():
                    rec_clean = net_G3(noisy)
                    test_psnr += util.calculate_psnr(util.tensor2uint(rec_clean), util.tensor2uint(clean))

            test_psnr /= test_num
            test_message['test_psnr'] = test_psnr
            message = '<epoch:{:d}> test_psnr: {:.3e}'.format(epoch, test_psnr)
            print(message)
            logger.info(message)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f'Epoch {epoch} finished in {epoch_duration:.2f} seconds- Train Loss: {train_loss:.4f}')

            net_G3.train()

        # update learning rate
        lr_scheduler_G3.step()


        # save model
        if epoch % 10 == 0:
            checkpoint_G3 = {
                'epoch': epoch,
                'step': current_step,
                'model_state_dict': net_util.get_bare_model(net_G3).state_dict(),
                'optimizer_state_dict': optim_G3.state_dict(),
                'scheduler_state_dict': lr_scheduler_G3.state_dict()
            }
            torch.save(checkpoint_G3, os.path.join(dir_save_models, 'checkpoint_G3_step_{:08d}.pth'.format(current_step)))
            print('checkpoint_G3_step_{:08d}.pth'.format(current_step))


    # Calculate and log total training time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f'Total training finished in {total_duration:.2f} seconds')    