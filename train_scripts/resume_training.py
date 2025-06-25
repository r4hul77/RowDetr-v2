import argparse
import os
from run_dist import *
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import CheckpointLoader, load_checkpoint
from run_dist import *



def create_args():
    parser = argparse.ArgumentParser(description='Resume Training Script')
    parser.add_argument("--config", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/tusimple/deform/regnetx_008.tv2_in1k/Decoder-V-Final-Deg4-l3-s5-np2-256-poly10/20241111_212104.py",
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/tusimple/deform/regnetx_008.tv2_in1k/Decoder-V-Final-Deg4-l3-s5-np2-256-poly25/best_F1 @ 10_epoch_348.pth",
                        help="Path to the checkpoint file")
    
    parser.add_argument("--launcher", type=str,
                        default='pytorch',
                        help="Launcher for Distributed Training")
    parser.add_argument('--local-rank', type=int, default=0)
 
    return parser.parse_args()

def main():
    parser = create_args()
    cfg = Config.fromfile(parser.config)
    cfg.update(load_from=parser.checkpoint)
    cfg.update(max_epochs=1000)
    cfg.update(resume=True)
    cfg.update(launcher=parser.launcher)
    runner = Runner.from_cfg(cfg)
    runner.train()
if __name__ == "__main__":
    main()