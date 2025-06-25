import os
import argparse
import shutil
import cv2
import tqdm
class ResFolder:
    def __init__(self, resolution, root_path):
        self.path = f"{root_path}/{resolution[0]}x{resolution[1]}"
        self.resolution = resolution
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(f"{self.path}/images", exist_ok=True)
        os.makedirs(f"{self.path}/labels", exist_ok=True)
    def run(self, img_file, label_file):

        img_id = img_file.split("/")[-1].split(".")[0]
        shutil.copy(img_file, f"{self.path}/images/{img_id}.jpg")
        shutil.copy(label_file, f"{self.path}/labels/{img_id}.json")


dir = "/home/r4hul-lcl/Datasets/row-detection/test"

img_dir = f"{dir}/images"
label_dir = f"{dir}/labels"
res2Folder = {}

for file in tqdm.tqdm(os.listdir(img_dir)):
    if file.endswith(".jpg"):
        img = cv2.imread(f"{img_dir}/{file}")
        res = img.shape[:2]
        if res not in res2Folder:
            res2Folder[res] = ResFolder(res, dir)
        res2Folder[res].run(os.path.join(img_dir, file), os.path.join(label_dir, file.replace(".jpg", ".json")))
