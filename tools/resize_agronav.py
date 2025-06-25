import cv2
import os
import numpy as np
import argparse
import glob
from tqdm import tqdm
def create_args():
    parser = argparse.ArgumentParser(description='Resize Agronav Dataset')
    parser.add_argument('--data_dir', type=str, default='/home/r4hul-lcl/Projects/agronav/lineDetection/data/inference/output/visualize_test', help='Path to the data directory')
    parser.add_argument('--og_dir', type=str, default='/home/r4hul-lcl/Datasets/row-detection/test/images', help='Path to the images directory')
    parser.add_argument('--output_dir', type=str, default='/home/r4hul-lcl/Datasets/row-detection/1-29/agronav_resized', help='Path to the output directory')
    parser.add_argument('--size', type=int, default=512, help='Size of the input image')
    return parser.parse_args()

def main():
    args = create_args()
    data_dir = args.data_dir
    og_dir = args.og_dir
    output_dir = args.output_dir
    size = args.size
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_path in tqdm(glob.glob(os.path.join(data_dir, '*.jpg'))):
        img_agnav = cv2.imread(img_path)
        img_og = cv2.imread(os.path.join(og_dir, os.path.basename(img_path)))
        img_agnav = cv2.resize(img_agnav, (img_og.shape[1], img_og.shape[0]))
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img_agnav)

if __name__ == "__main__":
    main()  