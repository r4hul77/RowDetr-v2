# This is a pythons script which takes in the dataset folder arguments and onnx file path.
# Loads the onnx file and runs the inference on the dataset folder
# saves the results in an output folder. 
import sys
sys.path.append('/home/r4hul-lcl/Projects/RowDetr')
from datasets.row_detection_dataset import RowDetectionDataset
import numpy as np
import cv2
import os
import argparse
from mmengine.registry import DATASETS, TRANSFORMS
from transforms.custom_transforms import *
import tqdm
from tools.infer_utils import *

def create_args():
    parser = argparse.ArgumentParser(description='Infer Onnx Model')
    parser.add_argument('--data_dir', type=str, default='/home/r4hul-lcl/Datasets/RowDetection/terrasentia', help='Path to the data directory')
    parser.add_argument('--onnx', type=str, default='/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist-1-29/efficientnet_lite0.ra_in1k-t2/F1_@_5.onnx', help='Path to the onnx file')
    parser.add_argument('--output_dir', type=str, default='/home/r4hul-lcl/Datasets/RowDetection/terrasentia_output', help='Path to the output directory')
    parser.add_argument('--size', type=int, default=512, help='Size of the input image')
    parser.add_argument('--out_labels', type=bool, default=False)
    parser.add_argument('--postprocessor', type=str, default='rowdetr', help='type of post processor rowdetr or rowcol')
    return parser.parse_args()

def main():
    args = create_args()
    data_dir = args.data_dir
    onnx_path = args.onnx
    output_dir = args.output_dir
    if args.postprocessor == 'rowdetr':
        print("RowDetr")
        postprocessor = PostProcessor(True, True, 10, True, False)
    elif args.postprocessor == 'rowcol':
        print("RowCol")
        postprocessor = RowColPostProcessor(True, True, 10, True, False)
    else:
        print(f"postprocessor {args.postprocessor} not found")
        return
    size = args.size
    pipe_line = [
        dict(type='LoadImageFromFile'),
        {"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
        {"type": 'Resize', "scale": (640, 384), "keep_ratio": False},
        #dict(type='LoadLabelFromFile'),
    ]
    pp_transformed = list(map(lambda x: TRANSFORMS.build(x), pipe_line))
    dataset = RowDetectionDataset(data_dir, pp_transformed)
    model = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if(args.out_labels):
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'imgs'), exist_ok=True)
    # postprocessor = PostProcessor(True, True, 10, True, False)
    # Now we will run the inference on the dataset
    for i in tqdm.tqdm(range(len(dataset))):
        data = dataset[i]
        img = data['img']
        img = np.expand_dims(img, axis=0)
        img = img.transpose(0, 3, 1, 2)
        img = img.astype(np.float32)
        polys, conf = model.run(None, {'input.1': img})
        img_id = data['img_path'].split('/')[-1].split('.')[0]
        if(args.out_labels):
            labels_path = os.path.join(os.path.join(output_dir, 'labels'), f'{img_id}.json')
        else:
            labels_path = None
        img_path = os.path.join(os.path.join(output_dir, 'imgs'), f'{img_id}.jpg')
        postprocessor.postprocess(polys, conf, data, label_path=labels_path, img_path=img_path)       
        # Save the results



if __name__ == '__main__':
    main()