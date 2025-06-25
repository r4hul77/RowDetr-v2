import argparse
import os
from train_scripts.run_dist import *
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import CheckpointLoader, load_checkpoint




def create_args():
    parser = argparse.ArgumentParser(description='Test Script For Row Detection')
    parser.add_argument('--data_dir', type=str, default='/home/r4hul-lcl/Datasets/row-detection/test', help='Path to the data directory')
    parser.add_argument("--config", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist-ablation/deform/regnetx_008.tv2_in1k/3-1-v-final-l3-higherlr/20241117_181343.py",
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-dist-ablation/deform/regnetx_008.tv2_in1k/3-1-v-final-l3-higherlr/best_F1 @ 10_epoch_430.pth",
                        help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="output-3-1", help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch Size For Testing")
    return parser.parse_args()

def main():
    parser = create_args()
    cfg = Config.fromfile(parser.config)
    rowDet = MODELS.build(cfg.model)
    N_DEGREE = cfg.model.n_degree
    W = 640
    H = 384
    test_dataloader = dict(
        batch_size=parser.batch_size,
        num_workers=4,
        sampler=dict(
            type="DefaultSampler",
            shuffle=False
        ),
        dataset=dict(
            type="RowDetection",
            dataset_dir=parser.data_dir,
            pipeline= [
                {"type": 'LoadImageFromFile'},
                {"type": 'Normalize', "mean": [123.65, 116.28, 103.936], "std": [58.395, 57.12, 57.375]},
                {"type": 'LoadLabelFromFile'},
                {"type": 'Resize', "scale": (W, H), "keep_ratio": False},
                {"type": "AddFirstKeyPoint", "normalized": True},
                {"type": 'CustomToTensor'},
                {"type": 'FitCurves', "n_degree": N_DEGREE, "normalize": True}
            ],
        ),
        collate_fn={"type": 'row_detection_metrics_collate'},
    )
    test_evaluator =  {
                        "type": "PolyDistanceEvaluator",
                        "metrics": [
                            {"type": "PolyDistanceMetric", "judging_loss_name": "PointWiseLoss", "threshold": 0.6},
                            {"type": "ClassicMetrics", "judging_loss_name": "PointWiseIOULoss", "threshold": 0.6, "area": True, "H": H, "W": W},
                           {"type": "TuSimpleMetric", "judging_loss_name": "PointWiseLoss", "threshold": 0.6, "positive_threshold": 0.75, "delta_x": 20},
                           {"type": "LPDMetric", "judging_loss_name": "PointWiseLoss", "H": H, "W": W}

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
    
    
