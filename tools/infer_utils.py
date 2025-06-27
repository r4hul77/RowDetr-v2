import pathlib
import cv2
import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
import logging
from PIL import Image
logger = logging.getLogger(__name__)
from mmengine.registry import MODELS
from mmengine.config import Config
from mmengine.runner import CheckpointLoader, load_checkpoint
import torch
import json
class ROSBagDataset:
    def __init__(self, rosbag_path, compressed_image_topic, camera_info_topic=None):
        super().__init__()
        self.dataset_path = pathlib.Path(rosbag_path)
        self.compressed_image_topic = compressed_image_topic
        self.camera_info_topic = camera_info_topic
        self.reader = AnyReader([Path(self.dataset_path)])
        self.reader.open()
        self.timestamps = []
        img_connections = [x for x in self.reader.connections if x.topic == self.compressed_image_topic]
        self.img_messages = self.reader.messages(connections=img_connections)
       # camera_info_connections = [x for x in self.reader.connections if x.topic == self.camera_info_topic]
    @staticmethod
    def CompressedImageMsg2Img(msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def __len__(self):
        count = self.reader.topics[self.compressed_image_topic].msgcount - 1
        logger.info(f"Number of images in {self.dataset_path}: {count}")
        return count

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        logger.info(f"Reading image {idx} from {self.dataset_path}")
        connection, timestamp, raw_msg = next(self.img_messages)
        msg = self.reader.deserialize(raw_msg, connection.msgtype)
        # if self.img_h == 0:
        #     self.img_h, self.img_w = msg.height, msg.width
        img = self.CompressedImageMsg2Img(msg)
        self.timestamps.append(timestamp/1e9)
        return img
    
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.read_img(i)
    
    def __del__(self):
        self.reader.close()
        
class OnnxInfer:
    def __init__(self, onnx_path, config, checkpoint):
        self.onnx_path = onnx_path
        self.model = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    def infer(self, img):
        return self.model.run(None, {'input.1': np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)})

class TorchInfer:
    def __init__(self, config, checkpoint):
        self.cfg = Config.fromfile(config)
        self.model = MODELS.build(self.cfg.model)
        self.model.eval()
        load_checkpoint(self.model, checkpoint)
        self.model.eval()
        self.model = self.model.cuda()
    def infer(self, img):
        with torch.no_grad():
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            polys, conf = self.model(img_tensor, mode="predict")
            return polys.cpu().numpy(), conf.cpu().numpy()

def get_infer(args):
    print("Args: ", args)
    if args.config is not None:
        print("Config: ", args.config)
        if args.checkpoint is not None:
            return TorchInfer(args.config, args.checkpoint)
    else:
        return OnnxInfer(args.onnx, args.config, args.checkpoint)

class PostProcessor:
    
    def __init__(self, labels_out, viz, number_points, draw_good_polys=True, draw_bad_polys=False, annotate_best_rows=True):
        self.labels_out = labels_out
        self.viz = viz
        self.number_of_points = number_points
        self.draw_good_polys = draw_good_polys
        self.draw_bad_polys = draw_bad_polys
        self.annotate_best_rows = annotate_best_rows
    @staticmethod
    def draw_polys(img, polys, color):    
        H, W, _ = img.shape
        lambdas = np.linspace(0, 1, 50)
        all_pnts = []
        for i in range(polys.shape[0]):
            poly = polys[i]
            x = poly[0, :]
            y = poly[1, :]
            x_t = np.polyval(x, lambdas)
            y_t = np.polyval(y, lambdas)
            x_t = x_t * W
            y_t = y_t * H
            x_t = x_t.astype(np.int32)
            y_t = y_t.astype(np.int32)
            pnts = np.vstack([x_t, y_t]).T
            all_pnts.append(pnts)
        
        for pnts in all_pnts:
            cv2.polylines(img, [pnts], isClosed=False, color=color, thickness=3)    
        return img
    
    def postprocess_viz(self, polys_all, conf, data=None):        
        if(data):
            if('img_path' in data.keys()):
                img = cv2.imread(data['img_path'])
            elif('img' in data.keys()):
                img = data['img']
        else:
            return
        img_out = img.copy()
        # now we will draw the original points
        good_polys = polys_all[0][conf[0] > 0.5, :, :]
        if self.draw_good_polys:
            img_out = PostProcessor.draw_polys(img_out, good_polys, (0, 0, 255))
        bad_polys = polys_all[0][conf[0] <= 0.5, :, :]
        
        if self.draw_bad_polys:
            img_out= PostProcessor.draw_polys(img_out, bad_polys, (255, 255, 255, 55))
        if self.annotate_best_rows:
            best_idxs = np.nonzero(np.where(conf[0] > 0.5, 1, 0))
            str2write = ""
            for idx in best_idxs:
                str2write += f"{idx} "
            cv2.putText(img_out, str2write, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if data:
            if("key_points" in data.keys()):
                key_points = data['key_points']
                for i, key_point in enumerate(key_points):
                    x = np.array(key_point['x'])
                    y = np.array(key_point['y'])
                    x = x.astype(np.int32)
                    y = y.astype(np.int32)
                    pnts = np.vstack([x, y]).T
                    cv2.polylines(img_out, [pnts], isClosed=False, color=(255, 0, 0), thickness=4)
        return img_out
    def make_labels(self, polys_all, conf, data=None):
        polys  = polys_all[0][conf[0] > 0.5, :, :]
        H, W, _ = cv2.imread(data['img_path']).shape
        lambdas = np.linspace(0, 1, self.number_of_points)
        ret_dict = {}
        if data:
            ret_dict['img_id'] = data['img_path'].split('/')[-1].split('.')[0]
        else:
            ret_dict['img_id'] = ''
        labels = []
        for i in range(polys.shape[0]):
            poly = polys[i]
            x = poly[0, :]
            y = poly[1, :]
            x_t = np.polyval(x, lambdas)
            y_t = np.polyval(y, lambdas)
            x_t = x_t * W
            y_t = y_t * H
            x_t = x_t.astype(np.int32)
            y_t = y_t.astype(np.int32)
            labels.append({
                'x': x_t.tolist(),
                'y': y_t.tolist(),
            })
        ret_dict['labels'] = labels
        return ret_dict

    def postprocess(self, polys, conf, data=None, label_path=None, img_path=None):
        if(self.viz):
            img = self.postprocess_viz(polys_all=polys, conf=conf, data=data)
            if(img_path):
                cv2.imwrite(img_path, img)
        if(self.labels_out):
            if(label_path):
                obj = self.make_labels(polys, conf, data)
                with open(label_path, 'w') as f:
                    json.dump(obj=obj, fp=f)

        
class RowColPostProcessor(PostProcessor):
    def __init__(self, labels_out, viz, number_points, draw_good_polys=True, draw_bad_polys=False):
        super().__init__(labels_out, viz, number_points, draw_good_polys, draw_bad_polys)
        print("RowColPostProcessor Initialized")
    def draw_poly(self, img, target, color, thickness, H, W):
        x = target[:-2]*W
        y = target[-2:]*H
        ys = np.linspace(np.max(y), np.min(y), 72)
        for j in range(1, len(x)):
            img = cv2.line(img, (int(x[j-1]), int(ys[j-1])), (int(x[j]), int(ys[j])), color, thickness) 
        return img
    def postprocess_viz(self, polys_all, conf, data=None):
        if(data):
            if('img_path' in data.keys()):
                img = cv2.imread(data['img_path'])
            elif('img' in data.keys()):
                img = data['img']
        else:
            return
        img_out = img.copy()
        H, W, C = img.shape
        good_polys = polys_all[0][conf[0] > 0.5]
        if self.draw_good_polys:
            for poly in good_polys:
                img_out = self.draw_poly(img_out, poly, (0, 0, 255), H=H, W=W, thickness=5)
        bad_polys = polys_all[0][conf[0] <= 0.5]
        if self.draw_bad_polys:
            img_out= self.draw_poly(img_out, bad_polys, (255, 255, 255, 55))
        return img_out
    def postprocess(self, polys, conf, data=None, label_path=None, img_path=None):
        return super().postprocess(polys, conf, data, label_path, img_path)
        
        
class ImgPostProcessor(PostProcessor):
    def __init__(self, labels_out, viz, number_points, draw_good_polys=True, draw_bad_polys=False):
        super().__init__(labels_out, viz, number_points, draw_good_polys, draw_bad_polys)
        print("ImgPostProcessor Initialized")
        
    def postprocess(self, polys, conf, data=None, label_path=None, img_path=None):
        return self.postprocess_viz(polys_all=polys, conf=conf, data=data)