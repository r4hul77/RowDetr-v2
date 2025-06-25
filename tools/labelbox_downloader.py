# This tool downloads the label data from labelbox and saves it in a local directory specified by the user.
from matplotlib import pyplot as plt
import labelbox
import os
import requests
import json
import io
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
class LabelBoxDownloader:
    def __init__(self, labelbox_api_key, project_id, output_dir):
        self.labelbox_api_key = labelbox_api_key
        self.project_id = project_id
        self.output_dir = output_dir
        # make directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = labelbox.Client(api_key=self.labelbox_api_key)
        self.x_init = []
        
    def download_labels(self):
        # This function downloads the label data from labelbox and saves it into memory.
        project = self.client.get_project(self.project_id)

        export_task = project.export_v2()
        export_task.wait_till_done()

        if(export_task.errors):
            print(export_task.errors)
            return
        return export_task.result
    @staticmethod
    def download_img(url):
        response = requests.get(url)
        bytes_im = io.BytesIO(response.content)
        return cv2.cvtColor(np.array(Image.open(bytes_im)), cv2.COLOR_RGB2BGR)

    def process_label(self, data_point):
        if(self.project_id not in data_point['projects']):
            return
        labels = data_point['projects'][self.project_id]["labels"]
    
        if(len(labels) == 0):
            return

        image_id = data_point['data_row']['id'] # this is the image id
        image_url = data_point['data_row']['row_data']
        objs = []
        for label in labels:
            objs += label['annotations']['objects']
        if(len(objs) == 0):
            return
        img = LabelBoxDownloader.download_img(image_url)
        return img, objs
    
    def filter_condition(self, label):
        if(self.project_id not in label['projects']):
            return False
        labels = label['projects'][self.project_id]["labels"]
        if(len(labels) == 0):
            return False
        objs = 0
        for label in labels:
            objs += len(label['annotations']['objects'])
            if(objs > 0):
                return True
        return False
    
    def format_label(self, objs, img_id=0):
        labels = []
        for obj in objs:
            obj_label = {
                'name': obj['name'],
            }
            obj_label['x'] = [point['x'] for point in obj['line']]
            obj_label['y'] = [point['y'] for point in obj['line']]
            pos = np.array([obj_label['x'], obj_label['y']])
            distances = np.cumsum(np.sqrt((np.diff(pos)**2).sum(axis=0)))
            para_distances = (distances/distances.max()).tolist()
            para_distances.insert(0, 0.0)
            obj_label['alpha'] = para_distances
            labels.append(obj_label)
        labels.sort(key=lambda x: x['x'][0])
        for i, label in enumerate(labels):
            label['name'] += f'_{i}'
        ret = {'img_id': img_id, 'labels': labels}
        return ret
    
    def save_labels(self, labels, split):
        dir = os.path.join(self.output_dir, split)
        os.makedirs(dir, exist_ok=True)
        imgs_dir = os.path.join(dir, "images")
        os.makedirs(imgs_dir, exist_ok=True)
        labels_dir = os.path.join(dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        x_inits = []
        for i, label in tqdm.tqdm(enumerate(labels)):
            img, objs = self.process_label(label)
            objs = self.format_label(objs, i)
            x_inits += [label['x'][0] for label in objs['labels']]
            
            if(img is None):
                continue
            img_path = os.path.join(imgs_dir, f"{i}.jpg")
            label_path = os.path.join(labels_dir, f"{i}.json")
            cv2.imwrite(img_path, img)
            with open(label_path, 'w') as f:
                objs = json.dumps(objs, indent=4)
                f.write(objs)
        self.make_hist(x_inits, 3, split)
        self.make_hist(x_inits, 4, split)
        self.make_hist(x_inits, 5, split)        

    def make_hist(self, x_inits, n, split):
        plt.figure()
        plt.hist(x_inits, n)
        plt.title(f'{split}_{n}')

    def run(self, train_test_sr, train_val_sr):
        print("Downloading labels...")
        labels = self.download_labels()
        print(f"{len(labels)} Labels Downloaded. Filtering unlabelled images...")
        labels_filtered = list(filter(self.filter_condition, labels))
        print(f"{len(labels_filtered)} Images with labels found. Splitting into train, test and val sets...")
        train, test = train_test_split(labels_filtered, test_size=train_test_sr)
        train, val = train_test_split(train, test_size=train_val_sr)
        print(f"Train: {len(train)} Test: {len(test)} Val: {len(val)}")
        print("Saving Train Labels...")
        self.save_labels(train, "train")
        print("Saving Test Labels...")
        self.save_labels(test, "test")
        print("Saving Val Labels...")
        self.save_labels(val, "val")
        plt.show()

            
    
if __name__ == "__main__":
    # load labelbox.api_key from environment variable
    
    with open('labelbox.api', 'r') as f:
        labelbox_api_key = f.read()
    #project_id = "cls3m6t2901ok07zb3fxe9rwq"
    #project_id = "cls3m6t2901ok07zb3fxe9rwq"
    project_id = "clyqfmdjo000q07zp3yxi4iyt"
    dir = "/home/r4hul/Datasets/row-detection-v6"
    downloader = LabelBoxDownloader(labelbox_api_key, project_id, dir)
    downloader.run(0.2, 0.2)