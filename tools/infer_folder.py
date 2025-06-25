import sys
sys.path.append('/home/r4hul-lcl/Projects/row_detection')
from datasets.row_detection_dataset import RowDetectionDataset
import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
from mmengine.registry import DATASETS, TRANSFORMS
from transforms.custom_transforms import *
import tqdm
from tools.infer_dataset import PostProcessor
from train_scripts.run_dist import *
from tools.infer_utils import *









def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default="/mnt/data/Datasets/RowDetection/terrasentia/images", required=False)
    parser.add_argument('--output_path', type=str, default="/mnt/data/Datasets/RowDetection/terrasentia_output", required=False)
    parser.add_argument('--onnx', type=str, default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist-1-29/efficientnet_lite0.ra_in1k-t2/F1_@_5.onnx", required=False)
    parser.add_argument('--config', type=str, default="", required=False)
    parser.add_argument('--checkpoint', type=str, default="", required=False)
    parser.add_argument('--video', type=bool, default=True, required=False)
    return parser.parse_args()




def main():
    args = create_args()
    folder_path = args.folder_path
    output_path = args.output_path
    onnx_path = args.onnx
    post_processor = ImgPostProcessor(labels_out=None, viz=False, number_points=None, draw_good_polys=True, draw_bad_polys=False)
    pipe_line = [
        {"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
        {"type": 'Resize', "scale": (640, 384), "keep_ratio": False},
        #dict(type='LoadLabelFromFile'),
    ]
    pp_transformed = list(map(lambda x: TRANSFORMS.build(x), pipe_line))
    if not args.config == "" and not args.checkpoint == "":
        model = TorchInfer(args.config, args.checkpoint)
    else:
        model = OnnxInfer(args.onnx, args.config, args.checkpoint)
    os.makedirs(output_path, exist_ok=True)
    
    data = {}
    if args.video:
        cap = None
    
    folder_list = os.listdir(folder_path)
    sorted_folder_list = sorted(folder_list, key=lambda x: int(x.split(".")[0]))
    for img_path in tqdm.tqdm(sorted_folder_list):
        img = cv2.imread(os.path.join(folder_path, img_path))
        data['img'] = img
        for transform in pp_transformed:
            data = transform(data)
        polys, conf = model.infer(data["img"])
        data['img'] = img
        img = post_processor.postprocess(polys, conf, data)
        base_name = os.path.basename(img_path)
        #cv2.imwrite(os.path.join(output_path, base_name), img)
        if args.video:
            if cap is None:
                H, W = img.shape[:2]
                cap = cv2.VideoWriter(os.path.join(output_path, "output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
            cap.write(img)
    if args.video:
        cap.release()


if __name__ == "__main__":
    main()
