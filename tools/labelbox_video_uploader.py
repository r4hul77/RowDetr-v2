import PIL.Image
import labelbox
import os
import cv2
import PIL
import math
import tqdm

class LabelBoxUploaderVideo:
    
    def __init__(self, labelbox_api_key, dataset_name, dir, prefix=""):
        self.labelbox_api_key = labelbox_api_key
        self.dataset_name = dataset_name
        self.output_dir = "temp"
        # make directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = labelbox.Client(api_key=self.labelbox_api_key)
        self.process_imgs(dir, prefix)
        self.upload(self.dataset_name)

        
        
    def process_imgs(self, dir, prefix=""):
        i = 0
        print("Processing Images")
        for root, dirs, files in tqdm.tqdm(os.walk(dir)):
            for file in files:
                if(file.endswith('mp4')):
                    cap = cv2.VideoCapture(os.path.join(root, file))
                    if not cap.isOpened():
                        print(f"Could not open {os.path.join(root, file)} file")
                        continue
                    else:
                        print(f"Processing Video File {os.path.join(root, file)}")
                        num = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if num%60 == 10:
                                cv2.imwrite(os.path.join(self.output_dir, f"{prefix}_{i}.jpg"), frame)
                                i += 1
                            num += 1
    
    
    
    def upload(self, dataset_name="testing123"):
        files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) \
            if os.path.isfile(os.path.join(self.output_dir, f))]
        limit = 15_000
        total_splits = int(math.ceil(len(files)/15_000.0))
        dataset = self.client.create_dataset(name=dataset_name)
        for i in range(total_splits):
            try:    
                task = dataset.create_data_rows(files[i*limit:(i+1)*limit])
                task.wait_till_done()
            except Exception as err:
                print(f'Error while creating labelbox dataset -  Error: {err}')

if __name__ == "__main__":
    with open('labelbox.api', 'r') as f:
        labelbox_api_key = f.read()
    project_id = "cls3m6t2901ok07zb3fxe9rwq"
    dir = "/mnt/data/row-detection-revanth"
    uploader = LabelBoxUploaderVideo(labelbox_api_key, "row_detection_rev", dir, "rev")
