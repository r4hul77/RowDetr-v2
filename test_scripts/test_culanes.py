import argparse
import os
from run_culanes import *
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import CheckpointLoader, load_checkpoint




def create_args():
    parser = argparse.ArgumentParser(description='Test Script For Row Detection')
    parser.add_argument('--data_dir', type=str, default='~/Datasets/row_detection/test', help='Path to the data directory')
    parser.add_argument("--config", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/culanes/deform/resnet50.a1_in1k/Decoder-V-Final-512/20241107_150201.py",
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/culanes/deform/resnet50.a1_in1k/Decoder-V-Final-512/best_F1 @ 10_epoch_85.pth",
                        help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch Size For Testing")
    return parser.parse_args()

def main():
    parser = create_args()
    cfg = Config.fromfile(parser.config)
    rowDet = MODELS.build(cfg.model)
    N_DEGREE = cfg.model.n_degree
    H = 384
    W = 640
    HOME = expanduser("~")
    test_dataloader = dict(
        batch_size=parser.batch_size,
        num_workers=4,
        sampler=dict(
            type="DefaultSampler",
            shuffle=False
        ),
        dataset={
                    "type": "CuLanes",
                    "txt_file": f"{HOME}/Datasets/lane-detection/culane/CULane/list/list/test.txt",
                    "dataset_dir": f"{HOME}/Datasets/lane-detection/culane/CULane",
                    "pipeline": [
                        {"type": 'LoadImageFromFile'},
                        {"type": 'Normalize', "mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5]},
                        {"type": 'LoadCuLanesLabelFromFile'},
                        {"type": 'Resize', "scale": (W, H), "keep_ratio": False},
                        {"type": 'CustomToTensor'},
                        {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": True}
                    ]
        },
        collate_fn={"type": 'row_detection_metrics_collate'},
    )
    test_evaluator =  {
                        "type": "PolyDistanceEvaluator",
                        "metrics": [
                            {"type": "PolyDistanceMetric", "judging_loss_name": "PointWiseLoss", "threshold": 0.5},
                            {"type": "ClassicMetrics", "judging_loss_name": "PointWiseLoss", "threshold": 0.5},
                            {"type": "TuSimpleMetric", "judging_loss_name": "PointWiseLoss", "threshold": 0.5, "H": H, "W": W},
                        #    {"type": "LPDMetric", "judging_loss_name": "PointWiseLoss", "H": H, "W": W}

                        ]
                    }
    cfg.update(test_dataloader=test_dataloader)
    cfg.update(test_evaluator=test_evaluator)
    cfg.update(work_dir=parser.output_dir)
    runner = Runner(model=rowDet, work_dir=parser.output_dir, load_from=parser.checkpoint, test_dataloader=test_dataloader, test_evaluator=test_evaluator, test_cfg= {
                        "type": 'TestLoop'
                    })
    runner.test()


if __name__ == "__main__":
    main()