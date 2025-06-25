import datasets.row_detection_dataset as row_detection_dataset
from models.poly_detection import PolyNet
# from mmengine.runner import Runner
from hooks.viz_hook import VizImageHook
from hooks.onnx_checkpointer import OnnxCheckpointer
from mmcv.transforms import LoadImageFromFile, Normalize, Resize
from transforms.custom_transforms import *
from metrics import *
from mmengine.evaluator import Evaluator
import warnings
from numpy.polynomial.polyutils import RankWarning 

warnings.simplefilter('ignore', RankWarning)
from mmengine.optim.scheduler import *
def main():
    NUM_WORKERS = 4
    BATCH_SIZE = 8*2
    N_DEGREE = 3
    N_POLY = 10
    custom_hooks = [
        dict(type="VizImageHook", interval=50, priority="VERY_HIGH")
    ]
    #Cosine Annealing Param Scheduler
    param_scheduler = [
        # Linear learning rate warm-up scheduler
        dict(type='LinearLR',
            start_factor=1e-6,
            by_epoch=True,  # Updated by iterations
            begin=0,
            end=5),  # Warm up for the first 50 iterations
        # The main LRScheduler
        dict(type='CosineAnnealingLR',
            by_epoch=True,  # Updated by epochs
            T_max=1000,
            eta_min=1e-5)
    ]
    
    train_dataloader = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=dict(
            type="DefaultSampler",
            shuffle=True
        ),
        dataset=dict(
            type="RowDetection",
            dataset_dir="/home/r4hul/Datasets/row-detection/train",
            pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                dict(type='LoadLabelFromFile'),
                dict(type='Resize', scale=(512, 512), keep_ratio=True),
                dict(type='CustomToTensor'),
                dict(type='CustomRandomFlip', prob=0.25),
                dict(type='Kansas', snow_prob=0.0, rain_prob=0.25),
                
                dict(type='CustomColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.25),
                dict(type='CustomRotate', limits=(-85, 85), p=0.75),
                dict(type='CustomMotionBlur', kernel_size=int(5), angle=(-0.5, 0.5), direction=(-1, 1), p=0.25),
                dict(type='CutOut', scale=(0.02, 0.3), prob=0.25, ratio=(0.3, 3.3)),
                dict(type='FitCurves', n_degree=N_DEGREE, normalize=True),            
            ]
        ),
        collate_fn={"type": 'row_detection_collate'},
    )
    
    val_dataloader = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=dict(
            type="DefaultSampler",
            shuffle=True
        ),
        dataset=dict(
            type="RowDetection",
            dataset_dir="/home/r4hul/Datasets/row-detection/val",
            pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                dict(type='LoadLabelFromFile'),
                dict(type='Resize', scale=(512, 512)),
                dict(type='CustomToTensor'),
                dict(type='FitCurves', n_degree=N_DEGREE, normalize=True),            
            ]
        ),
        collate_fn={"type": 'row_detection_collate'},
    )
    
    default_hooks = dict(
        best_checkpoint=dict(type='CheckpointHook', interval=1, save_best=['total_loss', 'accuracy', 'total_poly_loss'], rule=['less', 'greater', 'less'], max_keep_ckpts=5),
        checkpoint=dict(type='CheckpointHook', interval=5, save_last=True, max_keep_ckpts=5),
        )

    val_eval = PolyDistanceEvaluator(
        metrics=[
            dict(type="PolyDistanceMetric"),
        ]
    )
    
    cfg =  {
        "model": {
            "type": "PolyNet",
            "backbone": 'ecaresnet26t.ra2_in1k',
            "n_poly": N_POLY,
            "n_degree": N_DEGREE,
            "data_preprocessor": None,
            "class_coeff": 0.35,
            "poly_coeff": 0.35,
            "endpoint_coeff": 0.15,
            "reg_coeff": 0.15,
            "viz": False,
            "frozen": False
        },
        "train_dataloader": {
            "batch_size": BATCH_SIZE,
            "num_workers": 4,
            "sampler": {
                "type": "DefaultSampler",
                "shuffle": True
            },
            "dataset": {
                "type": "RowDetection",
                "dataset_dir": "/home/r4hul/Datasets/row-detection/train",
                "pipeline": [
                    {"type": 'LoadImageFromFile'},
                    {"type": 'Normalize', "mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5]},
                    {"type": 'LoadLabelFromFile'},
                    {"type": 'Resize', "scale": (512, 512), "keep_ratio": False},
                    {"type": 'CustomToTensor'},
                    {"type": 'CustomRandomFlip', "prob": 0.25},
                    {"type": 'Kansas', "snow_prob": 0.0, "rain_prob": 0.25},
                    {"type": 'CustomColorJitter', "brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.5, "p": 0.25},
                    {"type": 'CustomRotate', "limits": (-85, 85), "p": 0.75},
                    {"type": 'CustomMotionBlur', "kernel_size": 5, "angle": (-0.5, 0.5), "direction": (-1, 1), "p": 0.25},
                    {"type": 'CutOut', "scale": (0.02, 0.3), "prob": 0.25, "ratio": (0.3, 3.3)},
                    {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": True}
                ]
            },
            "collate_fn": {"type": 'row_detection_collate'},
        },
        "val_dataloader": {
            "batch_size": BATCH_SIZE,
            "num_workers": 4,
            "sampler": {
                "type": "DefaultSampler",
                "shuffle": True
            },
            "dataset": {
                "type": "RowDetection",
                "dataset_dir": "/home/r4hul/Datasets/row-detection/val",
                "pipeline": [
                    {"type": 'LoadImageFromFile'},
                    {"type": 'Normalize', "mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5]},
                    {"type": 'LoadLabelFromFile'},
                    {"type": 'Resize', "scale": (512, 512), "keep_ratio": False},
                    {"type": 'CustomToTensor'},
                    {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": True}
                ]
            },
            "collate_fn": {"type": 'row_detection_collate'},
        },
        "val_evaluator": {
            "type": "PolyDistanceEvaluator",
            "metrics": [
                {"type": "PolyDistanceMetric"}
            ]
        },
        "train_cfg": {
            "by_epoch": True,
            "max_epochs": 1000
        },
        "val_cfg": {
            "type": 'ValLoop'
        },
        "optim_wrapper": {
            "optimizer": {
                "type": "Adam",
                "lr": 1e-4
            }
        },
        "work_dir": "/home/r4hul/Projects/row_detection/results/ecaresnet26t.ra2_in1k",
        "visualizer": {
            "type": 'Visualizer',
            "vis_backends": [{"type": 'TensorboardVisBackend'}]
        },
        "custom_hooks": [
            {"type": "VizImageHook", "interval": 50, "priority": "VERY_HIGH"}
        ],
        "default_hooks": {
            "best_checkpoint": {"type": 'OnnxCheckpointer', "interval": 1, "save_best": ['total_loss', 'accuracy', 'total_poly_loss'], "rule": ['less', 'greater', 'less'], "max_keep_ckpts": 5},
            "checkpoint": {"type": 'OnnxCheckpointer', "interval": 5, "save_last": True, "max_keep_ckpts": 5}
        },
        "param_scheduler": [
            {"type": 'LinearLR', "start_factor": 1e-6, "by_epoch": True, "begin": 0, "end": 5},
            {"type": 'CosineAnnealingLR', "by_epoch": True, "T_max": 1000, "eta_min": 1e-5}
        ]
    }
    print(cfg)
    runner = Runner.from_cfg(cfg) 
    runner.train()

if __name__ == "__main__":
    main()
