import datasets.row_detection_dataset as row_detection_dataset
from mmengine.runner import Runner
#from blob_runner import OnnxRunner as Runner
from hooks.viz_row_col_hook import VizRowColHook
from hooks.onnx_checkpointer import OnnxCheckpointer
from mmcv.transforms import LoadImageFromFile, Normalize, Resize
from transforms.custom_transforms import *
from metrics.metrics import *
from models import *
from mmengine.evaluator import Evaluator
import warnings
#from mmengine.visualization import WandbVisBackend
from models.row_column_former.row_column_former import RowColumnFormer
from os.path import expanduser
from tools.export import save_best_checkpoints
import logging
from models.optmizers import AdEMAMix, AdEMAMixDistributedShampoo
logging.basicConfig(level=logging.INFO, filename="row_detection.log", filemode="w")
# Note this has Tanh Activation in the sampler head
warnings.simplefilter('ignore', np.RankWarning)
from mmengine.optim.scheduler import *
def main():
    HOME = expanduser("~")
    H = 384
    W = 640
    N = 72
    NUM_WORKERS = 30
    DEBUG = False
    BATCH_SIZE = 64
    if(DEBUG):
        BATCH_SIZE = BATCH_SIZE//2
    N_DEGREE = 3
    N_POLY = 12
    BACKBONES = [
        "resnet50.a1_in1k",
        # "resnet18.tv_in1k",
        # "efficientnet_lite0.ra_in1k",
        # "regnetx_008.tv2_in1k"
        ]
    WRK_DIR = "results/row_detection-row-col"
    sps = [2, 4]
    nps = [1, 2, 3]
    if(DEBUG):
        WRK_DIR = "results/row-col-former-debug"
    POST_FIX = "exact"
    MODEL_TYPE = 'RowColumnFormer'
    LOSS_LIST = [

        ([
            (
                {"type": "LineLocLoss"},
                0.6),
            (
                {"type": "LineIoULoss"},
                0.5
            ),
            ], "PointWiseLoss", False, "Decoder"),


    ]


    BACKBONE = "resnet50.a1_in1k"


    for val in LOSS_LIST:
        loss, metric, flip, head = val
        cfg =  {
            "model": {
                "H": H,
                "W": W,
                "type": MODEL_TYPE,
                "data_preprocessor": None,
                "polylossWithCoeffs": loss,
                "classLossesWithCoeffs": [({"type" : "BCEWithLogitsLoss"}, 0.5)],
                "viz": False,
                "d_model": 256,
                "nhead": 8,
                "layers": 4,
                "T": 3,
                "N": 72,
                "M": N_POLY,
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
                        {"type": "NormalizeRowDetectionLabel"},
                        {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": False},
                        {"type": "ConvertToPoints", "normalized": True, "N": 72, "H": H, "W": W}
                    ]
                },
                "collate_fn": {"type": 'row_col_former_collate'},
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
                        {"type": 'LoadRowDetectionLabel'},
                        #{"type": "AddFirstKeyPoint"},
                        {"type": 'CustomToTensor'},
                        {"type": 'CustomResize', "size": (W, H)},
                        {"type": "NormalizeRowDetectionLabel"},
                        {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": False},
                        {"type": "ConvertToPoints", "normalized": True, "N": 72, "H": H, "W": W}
                        
                    ]
                },
                "collate_fn": {"type": 'row_col_former_collate'},
            },
            "val_evaluator": {
                "type": "PolyDistanceEvaluator",
                "metrics": [
                    {"type": "RowColMetric", "threshold": 0.5},
                    # {"type": "ClassicMetrics", "judging_loss_name": metric, "threshold": 0.5, "H": H, "W": W},
                ]
            },
            "train_cfg": {
                "by_epoch": True,
                "max_epochs": 300,
            },
            "val_cfg": {
                "type": 'ValLoop'
            },
            "optim_wrapper": {
                "type": 'AmpOptimWrapper',
                "dtype": 'float16',
                "optimizer": {
                    "type": "AdamW",
                    "lr": 1e-4,
                },
                # "paramwise_cfg": {
                #     "backbone": dict(lr_mult=1.0),
                #     "encoder": dict(lr_mult=1.0),
                #     "poly_head": dict(lr_mult=0.1),
                # }
            },
            "work_dir": f"{WRK_DIR}/{BACKBONE}-{POST_FIX}",
            "visualizer": {
                "type": 'Visualizer',
                "vis_backends": [
                    {"type": 'TensorboardVisBackend'},
                    #{"type": "WandbVisBackend"}
                                ]
            },
            "custom_hooks": [
                    {"type": "VizRowColHook", "interval": 500, "N":N, "priority": "VERY_LOW"}
                ],
            "default_hooks": {
                "best_checkpoint": {"type": 'CheckpointHook', "interval": 1, "save_best": ['LineLocLoss_val', "LineIoULoss_val", "Mean Distance"], "rule": ['less', "less", "less"], "max_keep_ckpts": 5},
                "checkpoint": {"type": 'CheckpointHook', "interval": 5, "save_last": True, "max_keep_ckpts": 5}
            },
            # # "param_scheduler": [
            # #     {"type": 'LinearLR', "start_factor": 1e-11, "by_epoch": False, "begin": 0, "end":60},
            # #     {"type": 'CosineAnnealingLR', "by_epoch": True, "T_max": 1000, "eta_min": 1e-8, "begin": 5},
            #     #{"type": "MultiStepLR", "milestones": [300, 600, 900, 1100], "gamma": 0.5}
            # ],
            "launcher": "pytorch",
            "model_wrapper_cfg":dict(
                type='MMDistributedDataParallel', find_unused_parameters=True),
            
        }
        # if(not DEBUG):
        #     cfg["launcher"] = "pytorch"
        runner = Runner.from_cfg(cfg) 
        runner.train()
        import mmengine
        config = mmengine.config.Config(cfg_dict=cfg)
        save_best_checkpoints(metrics=['LineIoULoss_val', "LineLocLoss_val", "Mean Distance"], config=config, height=H, width=W, wrk_dir=cfg["work_dir"])
            
            
if __name__ == "__main__":
    main()