from sklearn.cluster import *
import json
import tqdm
import cv2
import numpy as np
import os
from labelbox_downloader import LabelBoxDownloader
from dataclasses import dataclass
import matplotlib.pyplot as plt

from cv2 import ximgproc

@dataclass
class RowDetectionParams:
    kernel_size: int = 25
    laplacian_kernel : int = 3
    
    


class crdlConvertor(LabelBoxDownloader):
    
    def __init__(self, crdl_dir, row_detection_params=RowDetectionParams()):
        self.output_dir = crdl_dir
        self.splits = []
        self.row_detection_params = row_detection_params
        for split in os.listdir(self.output_dir):
            if(os.path.isdir(os.path.join(self.output_dir, split))):
                self.splits.append(split)
        self.pipeline = [
            self.loadImg,
            self.gaussinBlur,
            self.padImage,
            self.laplacianFilter,
            self.removePadding
        ]
    def run(self):
        for split in self.splits:
            output_dir = os.path.join(self.output_dir, split, "labels") 
            os.makedirs(output_dir, exist_ok=True)
            label_dir = os.path.join(self.output_dir, split, "label")
            labels = os.listdir(label_dir)
            print("Saving labels for split: ", split)
            self.save_labels(labels, label_dir, output_dir)

    def process_label(self, data_point):
        # Converts the seg mask into a label of the form:
        # [{'name': 'row', 'line': [
            # {'x': x_cord, 'y': y_cord},
            # {'x': x_cord, 'y': y_cord},
            # ]}]
        labels = self.findRows(data_point)
        return labels

    def loadImg(self, img_path):
        img_og = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pix_skip = 10
        img_og[:pix_skip, :] = 0
        img_og[-pix_skip:, :] = 0
        img_og[:, :pix_skip] = 0
        img_og[:, -pix_skip:] = 0
        return img_og

    def gaussinBlur(self, img):
        return cv2.GaussianBlur(img, (self.row_detection_params.kernel_size, self.row_detection_params.kernel_size), 0)
    
    def padImage(self, img):
        return cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    
    def laplacianFilter(self, img):
        return cv2.Laplacian(img, cv2.CV_16S, ksize=self.row_detection_params.laplacian_kernel)
    def removePadding(self, img):
        return img[10:-10, 10:-10]
    
    # def findRows(self, data_point):

        
    #     # x, y = np.where(img == 255)
    #     # cords = np.array([x, y]).T
    #     # print(cords.shape)
    #     data_pnt = data_point
    #     img_og = self.pipeline[0](data_pnt)
    #     for func in self.pipeline:
    #         data_pnt = func(data_pnt)
        
    #     # clusters = DBSCAN(eps=1.5, min_samples=20).fit(cords)
    #     # total_colors = len(np.unique(clusters.labels_))
    #     # # Color the clusters by diving the colorspace (RGB) into total_colors, and assigning each cluster a color
    #     # colors = np.random.randint(0, 255, (total_colors, 3))
    #     # img_og = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     # for i, cord in enumerate(cords):
    #     #     img_og[cord[0], cord[1]] = colors[clusters.labels_[i]]
    #     # # write the number of clusters on the image
    #     # cv2.putText(img_og, f"Total Clusters: {total_colors}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     # return img_og
    #     abs_dst = cv2.convertScaleAbs(data_pnt)
    #     # Do edge detection
    #     # edges = cv2.Canny(img, 50, 150)
    #     # Find Lines
    #     lines = cv2.HoughLinesP(abs_dst, 1, np.pi/180, 100, minLineLength=50, maxLineGap=1)
    #     # Draw Lines
    #     #Convert image og into 3 channel
    #     img_og = cv2.cvtColor(abs_dst, cv2.COLOR_GRAY2BGR)
    #     slopes = []
    #     intercepts = []
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         slope = np.arctan2(y2 - y1, x2 - x1)
    #         intercept = y1 - slope * x1
    #         slopes.append(slope)
    #         intercepts.append(intercept)
    #     # plt slopes vs intercepts on matplotlib
    #     fig = plt.figure()
    #     fig.add_subplot(111)
    #     self.clusterer = DBSCAN(eps=0.001, min_samples=5)
    #     clusters = self.clusterer.fit(np.array([slopes]).T).labels_
    #     # Now plot the clusters in diffrent colors
    #     for cluster in np.unique(clusters):
    #         mask = clusters == cluster
    #         plt.scatter(np.array(slopes)[mask], np.array(intercepts)[mask], label=f'Cluster {cluster}')
    #     total_clusters = len(np.unique(clusters))
    #     colors = np.random.randint(0, 255, (total_clusters, 3))
    #     for i, line in enumerate(lines):
    #         if(clusters[i] == -1):
    #             continue
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(img_og, (x1, y1), (x2, y2), colors[clusters[i]].tolist(), 2)

    #     fig.canvas.draw()
    #     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     # now resize the data into the size of the image
    #     data = cv2.resize(data, (img_og.shape[1], img_og.shape[0]))
    #     # get the imaage of the plot
    #     #concatenate the image and the plot
    #     img_og = np.concatenate((img_og, data), axis=1)
    #     plt.close()
    #     return img_og
         
    
    def findRows(self, data_pnt, debug=False):
        img_og = self.pipeline[0](data_pnt)
        # Do connected componet analysis on img
        thinning = ximgproc.thinning(img_og)
        img_forlines = thinning
        img_forlines = cv2.dilate(thinning, np.ones((3, 3), np.uint8), iterations=1)
        lines = cv2.HoughLinesP(img_forlines, 1, np.pi/180, 50, minLineLength=50, maxLineGap=1)
        
        img_og = cv2.cvtColor(img_og, cv2.COLOR_GRAY2BGR)
        colors = np.random.randint(0, 255, (len(lines), 3))
        canvas = cv2.cvtColor(np.zeros_like(img_forlines), cv2.COLOR_GRAY2BGR)
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # Now do connected component analysis on the thinned image       
        img = cv2.dilate(thinning, np.ones((3, 3), np.uint8), iterations=1)
        
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        
        img = cv2.threshold(canvas_gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        num_labels, labels, stats, centriods = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
        
        colors = np.random.randint(0, 255, (num_labels, 3))

        zeros_color = np.zeros_like(img_og)
        rows = []
        for i in range(1, num_labels):
            zeros_color[labels==i] = colors[i]
            y, x = np.where(labels==i)
            rows.append({'name': f'row', 'line': [{'x': x_cord.astype(float), 'y': y_cord.astype(float)} for x_cord, y_cord in zip(x, y)]})
        img = np.concatenate((img_og, cv2.cvtColor(img_forlines, cv2.COLOR_GRAY2BGR), canvas, zeros_color), axis=1)
        if(debug):
            return img
        return rows
            
            
        
        

    def save_labels(self, labels, label_dir, output_dir):
        for i, label_name in tqdm.tqdm(enumerate(labels)):
            label = self.process_label(os.path.join(label_dir, label_name))
            num = label_name.split(".")[0]
            objs = self.format_label(label, int(num))
            label_path = os.path.join(output_dir, f"{num}.json")
            with open(label_path, 'w') as f:
                 objs = json.dumps(objs, indent=4)
                 f.write(objs)


def main():
    CRDLD_DATASET_PATH = "/home/r4hul/Datasets/CRDLD"
    crdlConvertor(CRDLD_DATASET_PATH).run()

if __name__ == "__main__":
    main()