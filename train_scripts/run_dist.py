import datasets.row_detection_dataset as row_detection_dataset
from models.poly_detection import PolyNet
from mmengine.runner import Runner
#from blob_runner import OnnxRunner as Runner
from hooks.viz_hook import VizImageHook
from hooks.onnx_checkpointer import OnnxCheckpointer
from mmcv.transforms import LoadImageFromFile, Normalize, Resize
from transforms.custom_transforms import *
from metrics.metrics import *
from models import *
from mmengine.evaluator import Evaluator
import warnings
#from mmengine.visualization import WandbVisBackend
from models.poly_transformer import NeuRowNet
from os.path import expanduser
from tools.export import save_best_checkpoints
import logging
from models.optmizers import AdEMAMix, AdEMAMixDistributedShampoo
logging.basicConfig(level=logging.INFO, filename="row_detection.log", filemode="w")
# Note this has Tanh Activation in the sampler head
from numpy.polynomial.polyutils import RankWarning 
# Suppress RankWarning
warnings.simplefilter('ignore', RankWarning)
from mmengine.optim.scheduler import *
import os
# logging.basicConfig(level=logging.INFO, filename="row_detection.log", filemode="w")
import torch
torch.manual_seed(4786)
def main():
    HOME = expanduser("~")
    H = 384
    W = 640
    NUM_WORKERS = 12
    DEBUG = False
    BATCH_SIZE = 32
    if(DEBUG):
        BATCH_SIZE = BATCH_SIZE//2
    N_POLY = 25
    BACKBONES = [
       # "resnet50.a1_in1k",
        #"resnet18.tv_in1k",
        #"efficientnet_lite0.ra_in1k",
        "regnetx_008.tv2_in1k"
        ]
    WRK_DIR = "results/row_detection-dist-6-24"
    os.makedirs(WRK_DIR, exist_ok=True)

    if(DEBUG):
        WRK_DIR = "results/row_detection-debug"
    MODEL_TYPE = 'NeuRowNet'
    LOSS_LIST = [
        #({"ChebyLoss": 0.35, "CircleLoss": 0.15}, "PointWiseOneOverLoss", True, "PolyOneOverHead"),
        # ("PointWiseOneOverLoss", "PointWiseOneOverLoss", True, "PolyOneOverHead"),
        # ("CircleLoss" ,"PointWiseOneOverLoss", True, "PolyOneOverHead"),
        ([
            (
                {"type": "PolyOptLoss"},
                1.0),

            ], "PointWiseLoss", False, "Decoder"),
        # ([
        #     (
        #         {"type": "RegLoss"},
        #         0.5)
        #     ], "PointWiseLoss", False, "Decoder"),

    ]
    custom_hooks = [
        dict(type="VizImageHook", interval=500, priority="VERY_HIGH")
    ]
    attention_type = "deform"
    val = LOSS_LIST[0]
    os.makedirs(os.path.join(WRK_DIR, attention_type), exist_ok=True)
    np = 2
    sp = 3
    # for N_DEGREE in N_DEGREEs:
    N_DEGREE = 2
    for BACKBONE in BACKBONES:
        loss, metric, flip, head = val
        POST_FIX = f"{BACKBONE}-t2"
        # check if POST_FIX is already in the WRK_DIR
        logging.info(f"Checking if {POST_FIX} exists")
        if(POST_FIX in os.listdir(os.path.join(WRK_DIR, attention_type))):
            print(f"Skipping {POST_FIX} because it already exists")
            logging.info(f"Skipping {POST_FIX} because it already exists")
            continue
        logging.info(f"Running {POST_FIX}")
        cfg =  {
            "model": {
                "size": (H, W),
                "type": MODEL_TYPE,
                "backbone": BACKBONE,
                "n_poly": N_POLY,
                "n_degree": N_DEGREE,
                "data_preprocessor": None,
                "polylossWithCoeffs": loss,
                "classLossesWithCoeffs": [({"type" : "BCEWithLogitsLoss"}, 0.25)],
                "viz": False,
                "frozen": False,
                "quant": True,
                "levels": 3,
                "head_levels": 3,
                "em_dim": 256,
                "sampling_points": sp,
                "num_points": np,
                "head": head,
                "num_heads": 16,
                "atten_type": attention_type,
                "use_poly_act": False,
                "pred_layers": 3,
                "deform_levels": 3,
            },
            "train_dataloader": {
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "sampler": {
                    "type": "DefaultSampler",
                    "shuffle": True
                },
                "dataset": {
                    "type": "RowDetection",
                    "dataset_dir": f"{HOME}/Datasets/row-detection/train",
                    "pipeline": [
                        {"type": 'LoadImageFromFile'},
                        {"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
                        {"type": 'LoadRowDetectionLabel'},
                        #{"type": "AddFirstKeyPoint"},
                        
                        {"type": 'CustomToTensor'},
                        {"type": 'CustomResize', "size": (W, H)},
                        {"type": 'CustomRandomAffine', "degrees": (-45, 45), "translate": (0.2, 0.2), "scale": (0.8, 1.5), "shear": 30, "p": 0.5},
                        {"type": 'Kansas', "snow_prob": 0.5, "rain_prob": 0.5},
                        {"type": 'CustomColorJitter', "brightness": 0.5, "contrast": 0.25, "saturation": 0.05, "hue": 0.15, "p": 0.5},
                        {"type": 'CustomMotionBlur', "kernel_size": 5, "angle": (-0.5, 0.5), "direction": (-1, 1), "p": 0.5},
                        {"type": 'CutOut', "scale": (0.1, 0.1), "prob": 0.5, "ratio": (0.3, 0.5)},
                        {"type": 'NormalizeRowDetectionLabel'},
                        {"type": 'CustomRandomFlip', "prob": 0.25},
                        {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": True}
                    ]
                },
                "collate_fn": {"type": 'row_detection_collate'},
            },
            "val_dataloader": {
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "sampler": {
                    "type": "DefaultSampler",
                    "shuffle": True
                },
                "dataset": {
                    "type": "RowDetection",
                    "dataset_dir": f"{HOME}/Datasets/row-detection/val",
                    "pipeline": [
                        {"type": 'LoadImageFromFile'},
                        {"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
                        {"type": 'LoadLabelFromFile'},
                        {"type": 'Resize', "scale": (W, H), "keep_ratio": False},
                        {"type": "AddFirstKeyPoint", "normalized": True},
                        {"type": 'CustomToTensor'},
                        {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": True}
                    ]
                },
                "collate_fn": {"type": 'row_detection_metrics_collate'},
            },
            "val_evaluator": {
                "type": "PolyDistanceEvaluator",
                "metrics": [
                    {"type": "PolyDistanceMetric", "judging_loss_name": "PointWiseLoss", "threshold": 0.5},
                    {"type": "ClassicMetrics", "judging_loss_name": metric, "threshold": 0.5, "H": H, "W": W},
                    {"type": "TuSimpleMetric", "judging_loss_name": "PointWiseLoss", "threshold": 0.6, "positive_threshold": 0.75, "delta_x": 20},
                    {"type": "LPDMetric", "judging_loss_name": "PointWiseLoss", "H": H, "W": W}

                ]
            },
            "train_cfg": {
                "by_epoch": True,
                "max_epochs": 10,
            },
            "val_cfg": {
                "type": 'ValLoop'
            },
            "optim_wrapper": {
                "type": 'AmpOptimWrapper',
                "dtype": 'float16',
                "optimizer": {
                    "type": "AdamW",
                    "lr": 2e-4,
                },
                "paramwise_cfg": {
                    "backbone": dict(lr_mult=0.1),
                    "encoder": dict(lr_mult=0.1),
                    "poly_head": dict(lr_mult=1.0),
                }
            },
            "work_dir": f"{WRK_DIR}/{POST_FIX}",
            "visualizer": {
                "type": 'Visualizer',
                "vis_backends": [
                    {"type": 'TensorboardVisBackend'},
                    #{"type": "WandbVisBackend"}
                                ]
            },
            # "custom_hooks": [
            #     {"type": "VizImageHook", "interval": 500, "flip":flip, "priority": "VERY_LOW"}
            # ],
            "default_hooks": {
                "best_checkpoint": {"type": 'CheckpointHook', "interval": 1, "save_best": ["F1 @ 10", "F1 @ 5"], "rule": ["greater", "greater"], "max_keep_ckpts": 5},
                "checkpoint": {"type": 'CheckpointHook', "interval": 5, "save_last": True, "max_keep_ckpts": 5}
            },
            "param_scheduler": [
                {"type": 'LinearLR', "start_factor": 1e-11, "by_epoch": False, "begin": 0, "end":60},
                {"type": 'CosineAnnealingLR', "by_epoch": True, "T_max":1200, "eta_min": 1e-11, "begin": 5},
                #{"type": "MultiStepLR", "milestones": [300, 600, 900, 1100], "gamma": 0.5}
            ],
            # "launcher": "pytorch",
            "model_wrapper_cfg":dict(
                type='MMDistributedDataParallel'),
            
        }
        if(not DEBUG):
            cfg["launcher"] = "pytorch"
        runner = Runner.from_cfg(cfg) 
        runner.train()
        import mmengine
        config = mmengine.config.Config(cfg_dict=cfg)
        save_best_checkpoints(metrics=['Mean PolyDistance', "F1 @ 10", "F1 @ 5", "Mean Side"], config=config, height=H, width=W, wrk_dir=cfg["work_dir"])
            
            
if __name__ == "__main__":
    main()
