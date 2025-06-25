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
import time




class ImgPostProcessor(PostProcessor):
    def __init__(self, labels_out, viz, number_points, draw_good_polys=True, draw_bad_polys=False):
        super().__init__(labels_out, viz, number_points, draw_good_polys, draw_bad_polys)
        print("ImgPostProcessor Initialized")
        
    def postprocess(self, polys, conf, data=None, label_path=None, img_path=None):
        return self.postprocess_viz(polys_all=polys, conf=conf, data=data)


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default="/mnt/data/Datasets/RowDetection/videos/IMG_0270.MOV", required=False)
    parser.add_argument('--output_path', type=str, default="/mnt/data/Datasets/RowDetection/output_videos/IMG_0270_efficientnet_lite0.ra_in1k-t2001v2.mp4", required=False)
    parser.add_argument('--onnx', type=str, default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist-1-29/regnetx_008.tv2_in1k-t2/F1_@_10.onnx", required=False)
    return parser.parse_args()
                

def main(args):
    start_time = time.time()
    video_path = args.video_path
    output_path = args.output_path
    onnx_path = args.onnx
    post_processor = ImgPostProcessor(labels_out=None, viz=False, number_points=None, draw_good_polys=True, draw_bad_polys=True)
    pipe_line = [
        {"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
        {"type": 'Resize', "scale": (640, 384), "keep_ratio": False},
        #dict(type='LoadLabelFromFile'),
    ]
    pp_transformed = list(map(lambda x: TRANSFORMS.build(x), pipe_line))
    model = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    cap = cv2.VideoCapture(video_path)
    # get first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video file")
        return
    H, W = frame.shape[:2]
    cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (W, H))
    data = {}
    i = 0
    while True:
        data['img'] = frame
        i += 1
        for transform in pp_transformed:
            data = transform(data)
        polys, conf = model.run(None, {'input.1': np.expand_dims(data['img'], axis=0).transpose(0, 3, 1, 2)})
        data['img']  = frame
        img = post_processor.postprocess(polys, conf, data)
        cap_out.write(img)
        ret, frame = cap.read()
        if not ret:
            break
    cap_out.release()
    cap.release()
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    print("FPS: ", i / (end_time - start_time))


if __name__ == "__main__":
    args = create_args()
    directory = "/mnt/data/Datasets/RowDetection/videos"
    output_directory = "/mnt/data/Datasets/RowDetection/output_videos/regnetxF10"
    os.makedirs(output_directory, exist_ok=True)
    for file in os.listdir(directory):
        print("processing video: ", file)
        args.video_path = os.path.join(directory, file)
        args.output_path = os.path.join(output_directory, file)
        main(args)
