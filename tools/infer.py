# This is a python script which infers a folder full of images using an onnx model.\


import os
import sys
sys.path.append('/home/r4hul-lcl/Projects/row_detection')
import PIL
import argparse
import numpy as np
import onnxruntime as ort
import cv2
import tqdm
from tools.infer_dataset import PostProcessor
from tools.infer_utils import *
from mmengine.registry import TRANSFORMS
from train_scripts.run_dist import *
def create_args():
    parser = argparse.ArgumentParser(description='Infer Onnx Model')
    parser.add_argument('--onnx', type=str, default='/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist/deform-256/regnetx_008.tv2_in1k/Decoder-v-final/F1_@_10.onnx', help='Path to the onnx file')
    parser.add_argument('--output_dir', type=str, default='regnetx008f1_10_reversed', help='Path to the output directory')
    parser.add_argument('--rosbag', type=str, default=None, help='Path to the rosbag file')
    parser.add_argument('--compressed_image_topic', type=str, default=None, help='Topic name of the compressed image')
    parser.add_argument('--camera_info_topic', type=str, default=None, help='Topic name of the camera info')
    parser.add_argument('--output_video', type=str, default=None, help='Path to the output video W.R.T the output directory')
    parser.add_argument('--config', type=str, default=None, help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Name of the model')
    return parser.parse_args()


def display_img_over_loop(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(root, file))
                cv2.imshow(file, img)
                cv2.waitKey(3000)
class Normalizer:
    
    def __init__(self, mean, stdev):
        self.mean = np.expand_dims(np.array(mean), axis=(0, 1))
        self.stdev = np.expand_dims(np.array(stdev), axis=(0, 1))
        
    def normalize(self, img):
        return (img - self.mean)/self.stdev
    


def main():
    args = create_args()
    output_dir = args.output_dir
    dataset = ROSBagDataset(args.rosbag, args.compressed_image_topic, args.camera_info_topic)
    
    pipe_line = [
        {"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
        {"type": 'Resize', "scale": (640, 384), "keep_ratio": False},
        #dict(type='LoadLabelFromFile'),
    ]
    pp_transformed = list(map(lambda x: TRANSFORMS.build(x), pipe_line))
    os.makedirs(output_dir, exist_ok=True)
    model = get_infer(args)
    i = 0
    post_processor = ImgPostProcessor(labels_out=None, viz=False, number_points=None, draw_good_polys=True, draw_bad_polys=False)
    data = {}
    if args.output_video:
        cap = None
    for img in tqdm.tqdm(dataset):
        data['img'] = img
        for transform in pp_transformed:
            data = transform(data)
        polys, conf = model.infer(data["img"])
        data['img'] = img
        img = post_processor.postprocess(polys, conf, data)
        #cv2.imwrite(os.path.join(output_path, base_name), img)
        if args.output_video:
            if cap is None:
                H, W = img.shape[:2]
                cap = cv2.VideoWriter(os.path.join(output_dir, args.output_video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
            cap.write(img)
    if args.output_video:
        cap.release()


if __name__ == '__main__':
    main()

    