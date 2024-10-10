# sRGB Real Noise Synthesizing with Neighboring Correlation-Aware Noise Model (CVPR'23)

paper [sRGB Real Noise Synthesizing with Neighboring Correlation-Aware Noise Model](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_sRGB_Real_Noise_Synthesizing_With_Neighboring_Correlation-Aware_Noise_Model_CVPR_2023_paper.pdf). 

## Prepare Dataset

1. Download SIDD-Medium sRGB part at [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php).
2. Use `Crop_SIDD.py` to crop images into 512×512 patches.
3. Run the code as

## Train 
### details 
- Without differentiating between cameras based on ISO level
    - ``--cam_name total``
    - You can check the data_preparetion folder to see how I split the train & test without separating the camera models based on ISO.
- if you use pretrain models
    - ``--use_pretrained``
    - ``--checkpoint_G3_name ~.pth``
```
!python main_train_neca_data.py --dir_save './saves/dncnn_neca_medium_total_aug/' --train_img_dir './dataset/SIDD_srgb_medium/Crop' --data_prep_dir './data_preparation/' --use_pretrained --checkpoint_G3_name checkpoint_G3_step_00000000.pth --not_aug --patch_size 96 --cam_name total --batch_size 32 --num_workers 4 --verbose --epochs 300 --gpu_ids 0 --test_epoch 5 --lr_decay_2 80 
```
