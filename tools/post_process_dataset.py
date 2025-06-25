import os
import shutil
import sys
sys.path.append('/home/r4hul-lcl/Projects/row_detection')
import json
import argparse
from transforms.custom_transforms import *
import cv2

def process_dataset(input):
    labels_dir = os.path.join(input, 'labels')
    images_dir = os.path.join(input, 'images')
    
    #move labels folder to labels_og
    labels_og_dir = os.path.join(input, 'labels_og')
    if os.path.exists(labels_og_dir):
        shutil.rmtree(labels_og_dir)
    shutil.move(labels_dir, labels_og_dir)
    load_label   = LoadLabelFromFile()
    add_first_key_point = AddFirstKeyPoint()
    #create new labels folder
    os.makedirs(labels_dir)
    for label_file in os.listdir(labels_og_dir):
        label_path = os.path.join(labels_og_dir, label_file)
        results = {}
        results["label_path"] = label_path
        results["ori_shape"] = cv2.imread(os.path.join(images_dir, label_file.replace('.json', '.jpg'))).shape
        results = load_label.transform(results)
        results = add_first_key_point.transform(results)
        for target in results["targets"]:
            target['x'] = (results['ori_shape'][1]*target['x']).tolist()
            target['y'] = (results['ori_shape'][0]*target['y']).tolist()
        # load json on to new label
        with open(label_path, 'r') as f:
            label = json.load(f)
        label['labels'] = results['targets']
        with open(os.path.join(labels_dir, label_file), 'w') as f:
            json.dump(label, f)
    print('Done')
        

    

def main():
    argparse.ArgumentParser(description='Post process dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input dataset')
    
    args = parser.parse_args()
    
    process_dataset(args.input)

if __name__ == "__main__":
    main()