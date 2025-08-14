import json
from typing import Dict, List, Tuple
from mmcv.transforms import BaseTransform, TRANSFORMS
import numpy as np
import kornia as kn
import torch
import cv2
@TRANSFORMS.register_module()
class LoadLabelFromFile(BaseTransform):
    
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, results):
        with open(results['label_path'], 'r') as f:
            label = json.load(f)
        results['key_points'] = label['labels']
        results["targets"] = []
        for target in results['key_points']:
            x = np.array(target['x'])/results['ori_shape'][1]
            y = np.array(target['y'])/results['ori_shape'][0]
            ret = {}
            ret['x'] = x
            ret['y'] = y
            results["targets"].append(ret)
        return results
@TRANSFORMS.register_module()
class LoadRowDetectionLabel(BaseTransform):
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, results):
        with open(results['label_path'], 'r') as f:
            label = json.load(f)
        results['key_points'] = label['labels']
        results["targets"] = []
        for target in results['key_points']:
            x = np.array(target['x'])
            y = np.array(target['y'])
            ret = {}
            ret['x'] = x
            ret['y'] = y
            results["targets"].append(ret)
        return results
@TRANSFORMS.register_module()
class NormalizeRowDetectionLabel(BaseTransform):
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, results):
        C, H, W = results['img'].shape
        for i, target in enumerate(results['targets']):
            results['targets'][i]['x'] = np.array(target['x'])/W
            results['targets'][i]['y'] = np.array(target['y'])/H
        return results

@TRANSFORMS.register_module()
class ConvertToPoints(BaseTransform):
    def __init__(self, normalized, N, H, W, DEG=2) -> None:
        super().__init__()
        self.normalized = normalized
        self.N = N
        self.H = H
        self.W = W
        self.DEG = DEG
    
    def transform(self, results):
        C, H, W = results['img'].shape
        n_lambdas = np.linspace(0, 1, self.N)
        temp_res = []
        for i in range(results['targets'].shape[0]):
            x_p = np.polyval(results['targets'][i, 0, :], n_lambdas)
            y_p = np.polyval(results['targets'][i, 1, :], n_lambdas)
            y_lambda_1 = np.polyfit(n_lambdas, y_p, 1)
            y_p = np.polyval(y_lambda_1, n_lambdas)
            temp_res.append(np.concatenate([x_p, y_p[:1], y_p[-1:]]))
        results['targets'] = np.array(temp_res)
        return results

@TRANSFORMS.register_module()
class LoadCuLanesLabelFromFile(BaseTransform):
    
    def __init__(self, normalize=True) -> None:
        super().__init__()
        self.normalize = normalize
    
    def transform(self, results):
        with open(results['label_path'], 'r') as f:
            lines = f.readlines()
        key_points = []
        results["targets"] = []
        for line in lines:
            x_y = line.rstrip().split(' ')
            x = [float(x) for x in x_y[::2]]
            y = [float(y) for y in x_y[1::2]]
            ret = {}
            if(self.normalize):
                ret['x'] = np.array(x)/results['ori_shape'][1]
                ret['y'] = np.array(y)/results['ori_shape'][0]
            else:
                ret['x'] = np.array(x)
                ret['y'] = np.array(y)
            results['targets'].append(ret)
            kp = {'x' : x, 'y': y}
            key_points.append(kp)
        results["key_points"] = key_points
        return results
            
@TRANSFORMS.register_module()
class LoadTusimpleLabel(BaseTransform):
    
    def __init__(self, normalize=True) -> None:
        super().__init__()
        self.normalize = normalize
    
    def transform(self, results):
        y = np.array(results['h_samples'])/results['ori_shape'][0]
        key_points = []
        results["targets"] = []
        for lane in results["lanes"]:
            x = np.array(lane)
            y_now = y[x>=0]
            x_now = x[x>=0]
            ret = {}
            if(self.normalize):
                ret['x'] = x_now/results['ori_shape'][1]
                ret['y'] = y_now
            else:
                ret['x'] = x_now
                ret['y'] =  np.array(results['h_samples'])[x>=0]
            results['targets'].append(ret)
            kp = {'x' : x_now, 'y': np.array(results['h_samples'])[x>=0]}
            key_points.append(kp)
        results["key_points"] = key_points
        return results        



@TRANSFORMS.register_module()
class AddFirstKeyPoint(BaseTransform):
    
    def __init__(self, normalized=True) -> None:
        super().__init__()
        if(normalized):
            self.x_max = 1.0
            self.y_max = 1.0
        self.normalized = normalized
    def compute_max(self, img):
        C, H, W = img.shape
        if(not self.normalized):
            self.x_max = W
            self.y_max = H
    
    def transform(self, results):
        self.compute_max(results['img'])
        for i, target in enumerate(results['targets']):
            x = np.array(target['x'])
            y = np.array(target['y'])
            y_x = np.polyfit(y, x, 1)
            Y = self.y_max
            X = np.polyval(y_x, Y)
            if(X < 0 or X > self.x_max):

                if(X > 0):
                    X = self.x_max
                else:
                    X = 0.
                x_y = np.polyfit(x, y, 1)
                Y = np.polyval(x_y, X)
            distance_from_frst = np.sqrt((X-x[0])**2 + (Y-y[0])**2)
            distance_from_end = np.sqrt((X-x[-1])**2 + (Y-y[-1])**2)
            if(distance_from_frst < distance_from_end):
                results['targets'][i]['x'] = np.insert(x, 0, X)
                results['targets'][i]['y'] = np.insert(y, 0, Y)
            else:
                results['targets'][i]['x'] = np.flip(np.append(x, X)).copy()
                results['targets'][i]['y'] = np.flip(np.append(y, Y)).copy()
        return results

@TRANSFORMS.register_module()
class FitCurves(BaseTransform):
    
    def __init__(self, n_degree, normalize=True): # normalize is not used
        super().__init__()
        self.n_degree = n_degree
        self.normalize = normalize
    
    def transform(self, results: Dict):
        targets_list = []
        for target in results['targets']:
            x_alpha, y_alpha = self.process_target(target, results['ori_shape'])
            if(x_alpha.size == 0):
                continue
            targets_list.append([x_alpha, y_alpha])
        if(len(targets_list) == 0):
            targets_list.extend([[np.array([]), np.array([])], [np.array([]), np.array([])]])
        results['targets'] = np.stack(targets_list, axis=2).transpose(2, 0, 1)
        return results

    def process_target(self, target, ori_shape):
        x =  np.array(target['x'], dtype=np.float32)
        y = np.array(target['y'], dtype=np.float32)

        # Temp Fix 
        #y = np.clip(y, 0.001, 1)

        x_alpha, y_alpha = self.fit_curves(x, y)
        return np.vstack((x_alpha, y_alpha))
    
    
    def fit_curves(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x = x[np.logical_and(y>=0, y<=1)]
        y = y[np.logical_and(y>=0, y<=1)]
        y = y[np.logical_and(x>=0, x<=1)]
        x = x[np.logical_and(x>=0, x<=1)]
        if(x.size < 3):
            return np.array([]), np.array([])
        dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        if(np.sum(dists) == 0):
            return np.array([]), np.array([])
        alpha = np.append([0.], np.cumsum(dists)/np.sum(dists))
        alpha = alpha.astype(np.float32)
        x_alpha = np.polyfit(alpha, x, self.n_degree)
        y_alpha = np.polyfit(alpha, y, self.n_degree)
        return x_alpha, y_alpha

@TRANSFORMS.register_module()
class Kansas(BaseTransform):
    
    def __init__(self, snow_prob, rain_prob) -> None:
        super().__init__()
        self.snow = kn.augmentation.RandomSnow(p=snow_prob)
        self.rain = kn.augmentation.RandomRain(p=rain_prob, drop_height=(1, 3), drop_width=(1, 3), number_of_drops=(50, 100))
    
    def transform(self, results):
        img = results['img'].unsqueeze(0)
        img = self.snow(img)
        results['img'] = self.rain(img).squeeze(0)
        return results

@TRANSFORMS.register_module()
class CutOut(BaseTransform):
    
    def __init__(self, scale, prob, ratio) -> None:
        super().__init__()
        self.cutout = kn.augmentation.RandomErasing(scale=scale, ratio=ratio, p=prob)
    
    def transform(self, results):
        results['img'] = self.cutout(results['img']).squeeze(0)
        return results

@TRANSFORMS.register_module()
class CustomRandomFlip(BaseTransform): #Works only for Normalized Coordinates
    
    def __init__(self, prob) -> None:
        super().__init__()
        self.prob = prob
    
    def transform(self, results):
        if(np.random.rand() < self.prob):
            H,W = results['ori_shape']
            results['img'] = kn.geometry.hflip(results['img'])
            for i, target in enumerate(results['targets']):
                results['targets'][i]['x'] = [1.0 - x for x in target['x']]
        return results

@TRANSFORMS.register_module()
class CustomToTensor(BaseTransform):
    
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, results):
        results['img'] = kn.image_to_tensor(results['img'])
        return results

@TRANSFORMS.register_module()
class CustomColorJitter(BaseTransform):
    
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.25) -> None:
        super().__init__()
        self.color_jitter = kn.augmentation.ColorJitter(brightness, contrast, saturation, hue, p=p)
    
    def transform(self, results):
        img = results['img'].unsqueeze(0)
        results['img'] = self.color_jitter(results['img']).squeeze(0)
        return results

@TRANSFORMS.register_module()
class CustomMotionBlur(BaseTransform):
    
    def __init__(self, kernel_size, angle, direction, p) -> None:
        super().__init__()
        self.motion_blur = kn.augmentation.RandomMotionBlur(p=p, kernel_size=kernel_size, angle=angle, direction=direction)
    
    def transform(self, results):
        img = results['img'].unsqueeze(0)
        results['img'] = self.motion_blur(img).squeeze(0)
        return results

@TRANSFORMS.register_module()
class CustomRotate(BaseTransform):
    
    def __init__(self, limits, p):
        super().__init__()
        self.limit_min = limits[0]
        self.limit_max = limits[1]
        self.p = p
    
    def transform(self, results):
        if(np.random.rand() < self.p):
            angle = torch.rand(1) * (self.limit_max - self.limit_min) + self.limit_min
            results['img'] = kn.geometry.rotate(results['img'].unsqueeze(0), angle).squeeze(0)
            center = torch.tensor([[0.5, 0.5]])
            scale = torch.ones((1, 2))
            matrix = kn.geometry.get_rotation_matrix2d(center=center, angle=angle, scale=scale)
            
            for i, target in enumerate(results['targets']):
                # Rotate the points
                pnts = torch.concat([torch.tensor(target['x']).unsqueeze(0), torch.tensor(target['y']).unsqueeze(0), torch.ones((1, len(target['x'])))], dim=0)
                pnts = matrix @ pnts.float()
                results['targets'][i]['x'], results['targets'][i]['y'] = pnts[0, 0, :].tolist(), pnts[0, 1, :].tolist()
        
        return results

@TRANSFORMS.register_module()
class CustomResize(BaseTransform):
    
    def __init__(self, size) -> None: # size is a tuple (W, H)
        super().__init__()
        self.size = size
    
    def transform(self, results):
        _, H, W = results['img'].shape
        results['ori_shape'] = (H, W)
        results['img'] = kn.geometry.resize(results['img'], (self.size[1], self.size[0]))
        w_scale = self.size[0]/results['ori_shape'][1]
        h_scale = self.size[1]/results['ori_shape'][0]
        for i, target in enumerate(results['targets']):
            results['targets'][i]['x'] = [x*w_scale for x in target['x']]
            results['targets'][i]['y'] = [y*h_scale for y in target['y']]
        return results
    
@TRANSFORMS.register_module()
class CustomRandomCameraParam(BaseTransform):
    def __init__(self, center_x, center_y, gamma, p) -> None:
        super().__init__()
        center_x = torch.tensor(center_x)
        center_y = torch.tensor(center_y)
        gamma = torch.tensor(gamma)
        self.aug = kn.augmentation.RandomFisheye(p=p, center_x=center_x, center_y=center_y, gamma=gamma)
    
    def transform(self, results):
        results['img'] = self.aug(results['img']).squeeze(0)
        _, H, W = results['img'].shape
        params = self.aug._params
        if params["batch_prob"][0].item() == 0:
            return results
        center_x = params['center_x'][0].item()
        center_y = params['center_y'][0].item()
        gamma = params['gamma'][0].item()
        DIST_R = ((1-center_x)**2 + (1-center_y)**2)**0.5
        R_W = int((1/(1 + DIST_R**gamma)*0.5+0.5)*W)
        R_H = int((1/(1 + DIST_R**gamma)*0.5+0.5)*H)
        DIST_S = ((-1-center_x)**2 + (-1-center_y )**2)**0.5
        S_W = int((-1/(1 + DIST_S**gamma)*0.5+0.5)*W)
        S_H = int((-1/(1 + DIST_S**gamma)*0.5+0.5)*H)
        # crop the image to the old size
        # results['img'] = results['img'][:, S_H:R_H, S_W:R_W]

        for i, target in enumerate(results['targets']):
            x = 2*(np.array(target['x'])/W - 0.5)
            y = 2*(np.array(target['y'])/H - 0.5)
            dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
            x = x/(1 + dist**gamma)
            y = y/(1 + dist**gamma)
            
            results['targets'][i]['x'] = (x*0.5 + 0.5)*W-S_W
            results['targets'][i]['y'] = (y*0.5 + 0.5)*H-S_H
        return results


@TRANSFORMS.register_module()
class CustomRandomAffine(BaseTransform):
    
    def __init__(self, degrees, translate, scale, shear, p) -> None:
        super().__init__()
        self.affine = kn.augmentation.RandomAffine(degrees, translate, scale, shear, p=p)
    
    def transform(self, results):
        results['img'] = self.affine(results['img']).squeeze(0)
        for i, target in enumerate(results['targets']):
            x = np.array(target['x'])
            y = np.array(target['y'])
            pnts = torch.stack([torch.tensor(x).unsqueeze(0), torch.tensor(y).unsqueeze(0), torch.ones((1, len(x)))], dim=1)
            pnts = self.affine.transform_matrix @ pnts.float()
            results['targets'][i]['x'] = pnts[0, 0, :].numpy()
            results['targets'][i]['y'] = pnts[0, 1, :].numpy()
        return results
