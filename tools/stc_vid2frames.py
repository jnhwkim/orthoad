import sys
import os
import numpy as np
import shutil
import argparse
import torch
import torchvision
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dataroot',
                        default='.',
                        help='Dataset root directory')
    parser.add_argument('--src_vid_path', default='archive/training/videos/',
                        help='Name of folder where `avi` files exist')
    parser.add_argument('--tar_vid_frame_path', default='converted/train',
                        help='Name of folder to save extracted frames.')
    parser.add_argument('--src_npy_path', default='archive/test_pixel_mask/',
                        help='Name of folder where `npy` frame mask exist')
    parser.add_argument('--tar_anno_path', default='converted/pixel_mask',
                        help='Name of folder to save extracted frame annotation')
    parser.add_argument('--extension', default='jpg',
                        help="File extension format for the output image")

    args = parser.parse_args()

    src_dir = os.path.join(args.dataroot, args.src_vid_path)
    tar_dir = os.path.join(args.dataroot, args.tar_vid_frame_path)

    try:
        os.makedirs(tar_dir)
    except FileExistsError:
        print(F'{tar_dir} already exists, remove whole tree and recompose ...')
        shutil.rmtree(tar_dir)
        os.makedirs(tar_dir)

    vid_list = os.listdir(src_dir)

    for i, vidname in enumerate(tqdm(vid_list)):
        vid = torchvision.io.read_video(os.path.join(src_dir, vidname), pts_unit='sec')[0]
        target_folder = os.path.join(tar_dir, vidname[:-4])
   
        try: 
            os.makedirs(target_folder)
        except FileExistsError:
            print(F'{target_folder} already exists, remove the directory recompose ...')
            shutil.rmtree(target_folder)
            os.makedirs(target_folder) 
            
        for i, frame in enumerate(vid):
            frame = (frame / 255.).permute(2, 0, 1) #HWC2CHW
            torchvision.utils.save_image(frame,
                                         F'{target_folder}/{i:03}.{args.extension}') 
    
    src_dir = os.path.join(args.dataroot, args.src_npy_path)    
    tar_dir = os.path.join(args.dataroot, args.tar_anno_path)

    try:
        os.makedirs(tar_dir)
    except FileExistsError:
        print(F"{tar_dir} already exists, remove whole tree and recompose ...")
        shutil.rmtree(tar_dir)
        os.makedirs(tar_dir)

    frame_anno = os.listdir(src_dir)

    for _f in tqdm(frame_anno):
        fn = _f[:-4]
        target_folder = os.path.join(tar_dir, fn)
        os.makedirs(target_folder)
        px_anno = np.load(F"{src_dir}/{fn}.npy").astype(np.float)

        for i, px_frame in enumerate(px_anno):
            torchvision.utils.save_image(torch.from_numpy(px_frame).unsqueeze(0), # CHW, 1 channel
                                         F"{target_folder}/{i:03}.{args.extension}")


if __name__ == '__main__':
    main()
