# Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images

paper [Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images](https://openaccess.thecvf.com/content/CVPR2022/papers/Maleky_Noise2NoiseFlow_Realistic_Camera_Noise_Modeling_Without_Clean_Images_CVPR_2022_paper.pdf). 

## Noise Synthesis Image

```
!git clone https://github.com/SamsungLabs/Noise2NoiseFlow.git
%cd /Noise2NoiseFlow/sRGB_noise_modeling
```
1. Place ``flow_srgb_synthesis.py``(Code for Noise Synthesis) in ./sRGB_noise_modeling
2. ``flow_srgb_synthesis.py`` assumes that you have not run ``train_noise_model.py``. If you ran ``train_noise_model.py``, the `logdir`, `sidd_path`, and `model_save_dir` in `/experiments/sidd/our_model/hps.txt` will be automatically replaced, but will not change if you have only run ``flow_srgb_synthesis.py``. If you have problems running it, please change the paths in `hps.txt`.
3. Modify the 413th line of code in `/Noise2NoiseFlow/data_loader/sidd_utils.py` to match your sidd medium dataset
### SIDD Medium
5. ``flow_srgb_synthesis_SIDD.py`` 
  - Modify the `--sidd_path`, `--model_save_dir` and `--synthesis_base_dir` paths to the your paths
```
!python flow_srgb_synthesis_SIDD.py --sidd_path '/home/gurwn/restoration/kaggle/dataset/SIDD_srgb_medium/Data' --model DnCNN_NM \
    --noise_model_path our_model --num_workers 4 --nm_load_epoch 60 --train_or_test 'train_dncnn' \
    --model_save_dir "/home/gurwn/restoration/kaggle/Noise2NoiseFlow/sRGB_noise_modeling/experiments/sidd/our_model/saved_models" \
    --synthesis_base_dir '/home/gurwn/restoration/kaggle/Noise2NoiseFlow/sRGB_noise_modeling/saved_noise_images/'  
```
### SIDD validation

```
!python flow_srgb_synthesis_nosidd.py --data_path '/home/gurwn/restoration/kaggle/dataset/SIDD_validation' --model DnCNN_NM \
    --noise_model_path our_model --num_workers 4 --nm_load_epoch 60 --train_or_test 'train_dncnn' \
    --model_save_dir "/home/gurwn/restoration/kaggle/Noise2NoiseFlow/sRGB_noise_modeling/experiments/sidd/our_model/saved_models" \
    --synthesis_base_dir '/home/gurwn/restoration/kaggle/Noise2NoiseFlow/sRGB_noise_modeling/synthesize'  
```
