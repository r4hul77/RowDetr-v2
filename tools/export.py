import sys
if 'mmengine.config' not in sys.modules:
    sys.path.append('/home/r4hul-lcl/Projects/row_detection')
    from run_dist import *
import onnx
from tools.onnx_custom_ops import *
from tools.to_onnx import convert2Onnx
import argparse
import os
from mmengine.runner import CheckpointLoader, load_checkpoint
from mmengine.config import Config
from mmengine.registry import MODELS
# check if run_dist is imported

def get_best_checkpoints(name2save_metric, wrk_dir):
    best_ckpts = {}
    metrics2names = {}
    for name, metric in name2save_metric.items():
        best_ckpts[name] = ''
        metrics2names[metric] = name
    
    for file in os.listdir(wrk_dir):
        if file.endswith(".pth"):
            for metric, name in metrics2names.items():
                if metric in file:
                    best_ckpts[name] = file
    return best_ckpts

def save_best_checkpoints(metrics, config, wrk_dir, width, height):
    names = [metric.replace(' ', '_') for metric in metrics]
    best_ckpts = get_best_checkpoints(dict(zip(names, metrics)), wrk_dir)
    for name, ckpt in best_ckpts.items():
        if ckpt != '':
            mmcv2Onnx(config=config, checkpoint_path=os.path.join(wrk_dir, ckpt), output_path=f"{wrk_dir}/{name}.onnx", width=width, height=height)
            print(f"Saved {name} to {wrk_dir}/{name}.onnx")
            

def mmcv2Onnx(config, checkpoint_path, output_path, width, height):

    def deploy(model):
        model.eval()
        for m in model.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return model    
    
    model = MODELS.build(config.model)
    model.eval()
    ckpt = load_checkpoint(model, checkpoint_path, map_location='cpu')

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = deploy(model)
        def forward(self, image):
            polys, outs = self.model._forward(image)
            return polys, outs.sigmoid()
        def eval(self):
            self.model.eval()
            
    wrapped = WrappedModel(model) 
    wrapped.model.eval()
    wrapped(torch.randn(1, 3, height, width))
    convert2Onnx(wrapped, [height, width], output_path)
def main():
    parser = argparse.ArgumentParser(description='Export Model To Onnx')
    parser.add_argument("--config", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist/deform/resnet50.a1_in1k/Decoder-640-384-Denoising-5-13-l4-hl3-pred3-deform3-s3-p3-nh-16/20241028_152307.py",
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist/deform/resnet50.a1_in1k/Decoder-640-384-Denoising-5-13-l4-hl3-pred3-deform3-s3-p3-nh-16/best_F1 @ 10_epoch_481.pth",
                        help="Path to the checkpoint file")
    parser.add_argument('--output',
                        type=str,
                        default='/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist/deform/resnet50.a1_in1k/Decoder-640-384-Denoising-5-13-l4-hl3-pred3-deform3-s3-p3-nh-16/F1_10.onnx',
                        help='Path to the output onnx file')
    parser.add_argument('--width',
                        type=int,
                        default=640,
                        help='Width of the input image')
    parser.add_argument('--height',
                        type=int,
                        default=384,
                        help='Height of the input image')
    
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    mmcv2Onnx(config=cfg, checkpoint_path=args.checkpoint, output_path=args.output, width=args.width, height=args.height)
if __name__ == "__main__":
    main()