import argparse
import os
from run_row_col_former import *
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import CheckpointLoader, load_checkpoint




def create_args():
    parser = argparse.ArgumentParser(description='Test Script For Row Detection')
    parser.add_argument('--data_dir', type=str, default='/home/r4hul-lcl/Datasets/row-detection/test', help='Path to the data directory')
    parser.add_argument("--config", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-row-col/resnet50.a1_in1k-exact/20241128_160559.py",
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/r4hul-lcl/Projects/row_detection/results/row_detection-row-col/resnet50.a1_in1k-exact/epoch_300.pth",
                        help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="output-3-1", help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size For Testing")
    return parser.parse_args()

def main():
    parser = create_args()
    cfg = Config.fromfile(parser.config)
    rowDet = MODELS.build(cfg.model)
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
                        {"type": 'LoadRowDetectionLabel'},
                        #{"type": "AddFirstKeyPoint"},
                        {"type": 'CustomToTensor'},
                        {"type": 'CustomResize', "size": (W, H)},
                        {"type": "NormalizeRowDetectionLabel"},
                        {"type": 'FitCurves', "n_degree": 3, "normalize": False},
                        {"type": "ConvertToPoints", "normalized": True, "N": 72, "H": H, "W": W}
            ],
        ),
        collate_fn={"type": 'row_detection_row_col_metrics_collate'},
    )
    test_evaluator =  {
                        "type": "PolyDistanceEvaluator",
                        "metrics": [
                           {"type": "TuSimpleMetricRowCol", "threshold": 0.6},
                           {"type": "LPDMetricRowCol", "threshold": 0.6}
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
    
    
