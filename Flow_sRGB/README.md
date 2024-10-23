# Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images

paper [Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images](https://openaccess.thecvf.com/content/CVPR2022/papers/Maleky_Noise2NoiseFlow_Realistic_Camera_Noise_Modeling_Without_Clean_Images_CVPR_2022_paper.pdf). 

## Noise Synthesis Image

```
!git clone https://github.com/SamsungLabs/Noise2NoiseFlow.git
%cd /Noise2NoiseFlow/sRGB_noise_modeling
```
1. Place ``flow_srgb_synthesis.py``(Code for Noise Synthesis) in ./sRGB_noise_modeling
2. ``flow_srgb_synthesis.py`` assumes that you have not run ``train_noise_model.py``. If you ran ``train_noise_model.py``, the `logdir`, `sidd_path`, and `model_save_dir` in `hps.txt` will be automatically replaced, but if you run `flow_srgb_synthesis.py` only, they will not be changed ``train_noise_model.py`` shouldn't be a problem, but if it is, please change the paths.
3. Modify the 413th line of code in `/Noise2NoiseFlow/data_loader/sidd_utils.py` to match your sidd medium dataset
4. ``flow_srgb_synthesis.py`` 
  - Modify the `--sidd_path`, `--model_save_dir` and `--synthesis_base_dir` paths to the your paths
```
!python flow_srgb_synthesis.py --sidd_path '/home/gurwn/restoration/kaggle/dataset/SIDD_srgb_medium/Data' --model DnCNN_NM \
    --noise_model_path our_model --num_workers 4 --nm_load_epoch 60 --train_or_test 'train_dncnn' \
    --model_save_dir "/home/gurwn/restoration/kaggle/Noise2NoiseFlow/sRGB_noise_modeling/experiments/sidd/our_model/saved_models" \
    --synthesis_base_dir '/home/gurwn/restoration/kaggle/Noise2NoiseFlow/sRGB_noise_modeling/saved_noise_images/'  
```
