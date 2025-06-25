from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS
import numpy as  np
import cv2
import torch
@HOOKS.register_module()
class VizRowColHook(Hook):
    def __init__(self, interval, N=72, **kwargs):
        self.interval = interval
        self.N = N
        self.val_it = 0
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        if runner.iter % self.interval == 0:
            self.visualize(runner, data_batch, "train")
            
    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        if self.val_it % self.interval == 0:
            print("In Viz Hook")
            self.visualize(runner, data_batch, "val")
        self.val_it += 1

    def visualize(self, runner, data_batch, mode):
        # This function visualizes the images and their predictions
        imgs = data_batch['images']
        runner.model.eval()
        with torch.no_grad():
            preds, cls_lgts = runner.model.forward(images=imgs.cuda(), mode="predict")
        runner.model.train()
        total_imgs = imgs.shape[0]
        imgs_list = list((imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)*127.5 + 127.5).astype(np.uint8))
        # Create a grid of images and their predictions
        imgs_preds = self.draw_preds(imgs_list, preds, (0, 255, 0), 2, cls_lgts)
        imgs_preds = self.draw_targets(imgs_preds, data_batch['targets'], (0, 0, 255), 2)
        # Log the grid to tensorboard
        imgs = np.array(imgs_preds)
        n = int(np.ceil(np.sqrt(total_imgs)))
        imgs_preds = np.concatenate([imgs, np.zeros((n**2-total_imgs, imgs.shape[1], imgs.shape[2], imgs.shape[3]))], axis=0)
        imgs_preds = imgs_preds.reshape((n, n, imgs.shape[1], imgs.shape[2], imgs.shape[3]), order='F')
        grid = np.concatenate(np.concatenate(imgs_preds, axis=2), axis=0).astype(np.uint8)
        if mode=='val':
            itr = self.val_it
        else:
            itr = runner.iter
        runner.visualizer.add_image(f"{mode}/preds", grid, itr)
            
    
    def draw_preds(self, imgs, polys, color, thickness, cls_logits):
        # This function draws the polys on the images
        grid = []
        for pnt in range(len(imgs)):
            img = imgs[pnt]
            W = img.shape[1]
            H = img.shape[0]
            img = np.ascontiguousarray(img)
            poly = polys[pnt, :, :].detach().cpu().numpy()
            for i in range(poly.shape[0]):
                if cls_logits[pnt, i] < 0.5:
                    continue
                    color= (255, 255, 255, 1)
                else:
                    color = (0, 255, 0)
                x = poly[i, :-2]*W
                y = poly[i, -2:]*H
                ys = np.linspace(np.max(y), np.min(y), self.N)
                for j in range(1, len(x)):
                    img = cv2.line(img, (int(x[j-1]), int(ys[j-1])), (int(x[j]), int(ys[j])), color, thickness)
                    #img = cv2.line(img, (int(x[j-1]), int(y1[j-1])), (int(x[j]), int(y[j])), (255, 0, 0), thickness)
            grid.append(img)
                        
        return grid

    def approx(self, roots, alpha1):
        roots_fix = roots.copy()
        roots_fix[roots<0] = -1 - roots[roots<0] - 1e-6
        roots_fix[roots>=0] = 1 + roots[roots>=0] + 1e-6
        alpha = roots_fix[0]
        beta = roots_fix[1]
        gamma = roots_fix[2]
    #-1/(a b c) 
    # - (x (a (b + c) + b c))/(a^2 b^2 c^2) 
    # - (x^2 (a^2 (b^2 + b c + c^2) + a b c (b + c) + b^2 c^2))/(a^3 b^3 c^3)
    # - (x^3 (a^3 (b^3 + b^2 c + b c^2 + c^3) + a^2 b c (b^2 + b c + c^2) + a b^2 c^2 (b + c) + b^3 c^3))/(a^4 b^4 c^4)
        d = -1/(alpha*beta*gamma)
        c = - ((alpha*(beta + gamma) + beta*gamma))/(alpha**2*beta**2*gamma**2) 
        b = - ((alpha**2*(beta**2 + beta*gamma + gamma**2) + alpha*beta*gamma*(beta + gamma) + beta**2*gamma**2))/(alpha**3*beta**3*gamma**3) 
        a = - ((alpha**3*(beta**3 + beta**2*gamma + beta*gamma**2 + gamma**3) + alpha**2*beta*gamma*(beta**2 + beta*gamma + gamma**2) + alpha*beta**2*gamma**2*(beta + gamma) + beta**3*gamma**3))/(alpha**4*beta**4*gamma**4)
        poly = np.stack([a, b, c, d], axis=-1)
        return np.polyval(poly, alpha1)
        
    def func(self, poly, alpha):
        oneOverY = np.polyval(poly, alpha)
        oneOverY[oneOverY==0] = 1e-6
        return 1/oneOverY

    def draw_targets(self, imgs, targets, color, thickness):
        grid = []
        for i in range(len(imgs)):
            img = imgs[i]
            W = img.shape[1]
            H = img.shape[0]
            target = targets[i].detach().cpu().numpy()
            alpha = np.linspace(0, 1, 10)
            no_polys = target.shape[0]
            img = np.ascontiguousarray(img)
            for i in range(no_polys):
                img = self.draw_poly(img, target[i], color, thickness, H, W)
            grid.append(img)
                        
        return grid
    
    def draw_poly(self, img, target, color, thickness, H, W):
        x = target[:-2]*W
        y = target[-2:]*H
        ys = np.linspace(np.max(y), np.min(y), self.N)
        for j in range(1, len(x)):
            img = cv2.line(img, (int(x[j-1]), int(ys[j-1])), (int(x[j]), int(ys[j])), color, thickness) 
        return img