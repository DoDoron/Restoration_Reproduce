import os
import utils.utils_image as util

def main():
    ori_dir = '/home/gurwn/restoration/kaggle/dataset/SIDD_srgb_medium/Data' # put original SIDD directory here
    target_dir = '/home/gurwn/restoration/kaggle/dataset/SIDD_srgb_medium/Crop' # put target cropped SIDD directory here

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    util.crop_sidd_dataset(ori_dir, target_dir, p_size=512, p_overlap=128, p_max=800)

if __name__ == '__main__':
    main()