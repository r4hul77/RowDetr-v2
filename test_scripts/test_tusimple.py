import argparse
import os
from run_tusimple import *
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import CheckpointLoader, load_checkpoint




def create_args():
    parser = argparse.ArgumentParser(description='Test Script For Row Detection')
    parser.add_argument('--data_dir', type=str, default='~/Datasets/row_detection/test', help='Path to the data directory')
    parser.add_argument("--config", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/tusimple/deform/regnetx_008.tv2_in1k/Decoder-V-Final-Deg4-l3-s5-np2-256-poly10/20241111_212104.py",   
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/tusimple/deform/regnetx_008.tv2_in1k/Decoder-V-Final-Deg4-l3-s5-np2-256-poly10/best_F1 @ 10_epoch_999.pth",
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
                            "type": "Tusimple",
                            "dataset_dir": f"{HOME}/Datasets/lane-detection/tusimple/TUSimple/test_set",
                            "pipeline": [
                                {"type": 'LoadImageFromFile'},
                                {"type": 'Normalize', "mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5]},
                                #{"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
                                {"type": 'LoadTusimpleLabel'},
                                #{"type": "AddFirstKeyPoint"},
                                {"type": 'Resize', "scale": (W, H), "keep_ratio": False},
                                {"type": 'CustomToTensor'},
                                {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": True}
                            ]
                        },
        collate_fn =  {"type": 'row_detection_metrics_collate'},
    )
    test_evaluator =  {
                        "type": "PolyDistanceEvaluator",
                        "metrics": [
                            {"type": "PolyDistanceMetric", "judging_loss_name": "PointWiseLoss", "threshold": 0.5},
                            {"type": "ClassicMetrics", "judging_loss_name": "PointWiseLoss", "threshold": 0.5},
                            {"type": "TuSimpleMetric", "judging_loss_name": "PointWiseLoss", "threshold": 0.5, "H": H, "W": W, "delta_x": 30, "positive_threshold": 0.85, "N_DEGREE": N_DEGREE},
                            {"type": "LPDMetric", "judging_loss_name": "PointWiseLoss", "H": H, "W": W, "N_DEGREE": N_DEGREE}

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