import sys
sys.path.append('/home/r4hul-lcl/Projects/row_detection')
import PIL.Image
import labelbox
import os

import PIL
import math
import tqdm
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import cv2
from tools.infer_utils import *
from mmengine.registry import TRANSFORMS
import argparse
from train_scripts.run_dist import *

def create_args():
    parser = argparse.ArgumentParser(description='Infer Onnx Model')
    parser.add_argument('--onnx', type=str, default='/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist/deform-256/regnetx_008.tv2_in1k/Decoder-v-final/F1_@_10.onnx', help='Path to the onnx file')
    parser.add_argument('--output_dir', type=str, default='regnetx008f1_10_reversed', help='Path to the output directory')
    parser.add_argument('--rosbag', type=str, default=None, help='Path to the rosbag file')
    parser.add_argument('--compressed_image_topic', type=str, default=None, help='Topic name of the compressed image')
    parser.add_argument('--camera_info_topic', type=str, default=None, help='Topic name of the camera info')
    parser.add_argument('--output_video', type=str, default=None, help='Path to the output video W.R.T the output directory')
    parser.add_argument('--config', type=str, default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist-1-25/regnetx_008.tv2_in1k-t2/20250127_235302.py", help='Path to the config file')
    parser.add_argument('--checkpoint', type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist-1-25/regnetx_008.tv2_in1k-t2/best_F1 @ 5_epoch_1199.pth",
                        help='Name of the model')
    return parser.parse_args()

def get_tiff_files(path):
    return [f for f in os.listdir(path) if f.endswith('.tif')]

def get_tiff_file_paths(path):
    return [os.path.join(path, f) for f in get_tiff_files(path)]

def get_tiff_file_paths_recursive(path):
    return [os.path.join(dirpath, filename)
            for dirpath, _, filenames in os.walk(path)
            for filename in filenames
            if filename.endswith('.tif')]


path = "/home/r4hul-lcl/Datasets/RowDetection/tiff_uav"
output_dir = "/home/r4hul-lcl/Datasets/RowDetection/tiff_uav_split"
pipe_line = [
        {"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
        {"type": 'Resize', "scale": (640, 384), "keep_ratio": False},
        #dict(type='LoadLabelFromFile'),
    ]
pp_transformed = list(map(lambda x: TRANSFORMS.build(x), pipe_line))
args = create_args()
model = get_infer(args)
print(args)
os.makedirs(output_dir, exist_ok=True)
post_processor = ImgPostProcessor(labels_out=None, viz=False, number_points=None, draw_good_polys=True, draw_bad_polys=False)

for file_path in get_tiff_file_paths_recursive(path):
    with rasterio.open(file_path) as src:
       img = src.read()
       H, W = img.shape[1:]
       img_color = np.clip(img[:3, :, :], 0, 8192)
       img_color[img_color > 2**12] = 0
       img_color = np.clip(img_color, 0, 2**12)
       # Draw histogram of the image for each channel
       plt.hist(img_color[0].flatten(), bins=2000, range=(0, 8192))
       plt.hist(img_color[1].flatten(), bins=2000, range=(0, 8192))
       plt.hist(img_color[2].flatten(), bins=2000, range=(0, 8192))
       plt.show()
       
       img_color = img_color[:, H//2-384//2:H//2+384//2, W//2-640//2:W//2+640//2]



       
       denom = img_color.max(axis=(1,2), keepdims=True) - img_color.min(axis=(1,2), keepdims=True)
       min_col = img_color.min(axis=(1,2), keepdims=True)
       img_color = (img_color - min_col) / denom * 255
       img_color = img_color.astype(np.uint8)
       img_color = img_color.transpose(2, 1, 0)
       # Convert to RGB
       data = {}
       data['img'] = img_color.copy()
       img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
       img_color = PIL.Image.fromarray(img_color)
       img_color.save(os.path.join("tiff_jpg.jpg"))
       for transform in pp_transformed:
           data = transform(data)
       polys, conf = model.infer(data["img"])
       data['img'] = np.array(img_color)
       img = post_processor.postprocess(polys, conf, data)
       cv2.imwrite(os.path.join("row_detections.jpg"), img)




