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

def create_args():
    parser = argparse.ArgumentParser(description='Infer Onnx Model')
    parser.add_argument('--data_dir', type=str, default='/home/r4hul-lcl/Datasets/row_detection_north_farm_9_24', help='Path to the data directory')
    parser.add_argument('--onnx', type=str, default='/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist/deform-256/regnetx_008.tv2_in1k/Decoder-v-final/F1_@_10.onnx', help='Path to the onnx file')
    parser.add_argument('--output_dir', type=str, default='regnetx008f1_10_reversed', help='Path to the output directory')
    
    return parser.parse_args()

def infer(ort_sess, img):
    img_resized = cv2.resize(img, (640, 384))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = img_resized.astype(np.float32)
    # img_resized = (img_resized - 127.5) / 127.5
    img_resized = img_resized.transpose(0, 3, 1, 2)
    ort_inputs = {ort_sess.get_inputs()[0].name: img_resized}
    ort_outs = ort_sess.run(None, ort_inputs)
    return ort_outs
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
    data_dir = args.data_dir
    onnx_file = args.onnx
    output_dir = args.output_dir
    
    
    mean =  [123.65, 116.28, 103.936]
    std =  [58.395, 57.12, 57.375]
    normalizer = Normalizer(mean=mean, stdev=std)
    os.makedirs(output_dir, exist_ok=True)
    ort_session = ort.InferenceSession(onnx_file)
    i = 0
    postprocessor = PostProcessor(True, True, 10, True, False)
    for root, dirs, files in tqdm.tqdm(os.walk(data_dir)):
        for file in tqdm.tqdm(files):
            if file.endswith('.tiff'):
                im = PIL.Image.open(os.path.join(root, file))
                im = im.convert("RGB")
                img = np.array(im)[:, :, ::-1]
                img_normalized = normalizer.normalize(img)
                polys, conf = infer(ort_session, img_normalized)
                postprocessor.postprocess(polys, conf, data={'img': img}, img_path = os.path.join(output_dir, f"{i}.jpg"))
                i += 1    
if __name__ == '__main__':
    main()

    