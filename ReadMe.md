# RowDetr-v2: End-to-End Crop Row Detection Using Polynomials

[![arXiv](https://img.shields.io/badge/arXiv-2412.10525-b31b1b.svg)](https://arxiv.org/abs/2412.10525)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.1016/j.atech.2025.101494)

## Introduction

RowDetr-v2 is an advanced implementation of the **RowDetr** framework for end-to-end crop row detection using polynomial representations. This repository provides the official code for the paper ["RowDetr: End-to-End Crop Row Detection Using Polynomials"](https://arxiv.org/abs/2412.10525), published in *Smart Agricultural Technology* (2025).

Crop row detection is essential for enabling autonomous robots to navigate in GPS-denied agricultural environments, particularly under the canopy where occlusions, gaps, and curved rows pose significant challenges to traditional vision-based methods. RowDetr-v2 addresses these limitations by leveraging attention mechanisms and polynomial-based modeling to achieve robust, post-processing-free detection.

Key features:
- **End-to-end detection**: No manual post-processing required.
- **Polynomial parameterization**: Represents crop rows as smooth polynomials for accurate curve fitting.
- **Attention-based architecture**: Utilizes transformer mechanisms for global context understanding.
- **GPS-denied robustness**: Optimized for under-canopy scenarios with heavy occlusions.

## Installation

### Prerequisites
- Python 3.11
- CUDA-enabled GPU (recommended for training/inference)
- Conda package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/r4hul77/RowDetr-v2.git
   cd RowDetr-v2
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -n RowDetr python=3.11
   conda activate RowDetr
   ```

3. Install dependencies:
   ```bash
   pip install mmengine
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   conda install nvidia/label/cuda-12.1.0::cuda-toolkit
   pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
   pip install timm
   pip install scipy
   pip install sortedcontainers
   pip install onnx
   pip install onnxscript
   pip install future tensorboard
   ```

4. (Optional) Install in development mode:
   ```bash
   pip install -e .
   ```

## Dataset

The implementation uses the **Crop Row Detection Dataset**, specifically collected and labeled to overcome challenges in under-canopy environments. The dataset includes images and JSON annotations for train/test/validation splits.

### Download
Download the dataset from Kaggle:
- [Crop Row Detection Dataset](https://www.kaggle.com/datasets/rahulharsha/crop-row-detection/)

### Structure
After downloading and extracting, the dataset follows this structure:
```
CropRowDetectionDataset/
├── Train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.json
│       └── ...
├── Test/
│   ├── images/
│   └── labels/
└── Validation/
    ├── images/
    └── labels/
```

### JSON Annotation Format
Each JSON file contains:
- `img_id`: Image identifier.
- `labels`: List of crop rows with `name`, `x`, `y` coordinates, and `alpha` (normalized distance values for polynomial parameterization).

Example:
```json
{
  "img_id": 0,
  "labels": [
    {
      "name": "row_0",
      "x": [530.843, 588.246, ...],
      "y": [1189.066, 1044.191, ...],
      "alpha": [0.0, 0.15029798560170535, ...]
    }
  ]
}
```

For detailed dataset description, see the [dataset README](dataset/README.md) or the [Kaggle page](https://www.kaggle.com/datasets/rahulharsha/crop-row-detection/).

## Usage

### Training
TODO

### Evaluation
TODO

### Inference
TODO

For more details, see [docs/USAGE.md](docs/USAGE.md).

## Model Architecture

RowDetr builds on DETR-like transformer architectures but specializes in:
- **Polynomial Head**: Outputs coefficients for row polynomials instead of bounding boxes.
- **Alpha Normalization**: Uses normalized distances (`alpha`) for consistent row parameterization.
- **Multi-Row Detection**: Handles multiple parallel crop rows in a single image.

See `models/rowdetr_v2.py` for the core implementation.

## Results

The RowDetr-v2 models, along with comparative baselines, achieve state-of-the-art performance on the Crop Row Detection Dataset. The table below summarizes the results:

| Model              | Latency (↓) | Param Count (↓) | LPD (↓) | TuSimple F1 (↑) | TuSimple FPR (↓) | TuSimple FNR (↓) |
|---------------------|-------------|-----------------|---------|-----------------|--------------------|-------------------|
| RowDetr[efficientnet] | 9.11 ms    | 23M            | 0.405   | 0.734           | 0.393             | 0.044             |
| RowDetr[resnet18]  | 6.7 ms     | 31M            | 0.421   | 0.736           | 0.391             | 0.043             |
| RowDetr[regnetx_008] | 9.7 ms   | 27M            | 0.416   | 0.725           | 0.404             | 0.046             |
| RowDetr[resnet50]  | 9.25 ms    | 44M            | 0.413   | 0.740           | 0.384             | 0.046             |
| Agronav            | 18 ms      | NA             | 0.825   | NA              | NA                | NA                |
| RowCol [12]        | 14.16 ms   | 35M            | 1.48    | 0.3191          | 0.8028            | 0.0400            |

- **Latency**: Inference time per image (lower is better).
- **Param Count**: Number of model parameters (lower is better).
- **LPD**: Lane Position Deviation (lower is better).
- **TuSimple F1**: F1 score on TuSimple dataset (higher is better).
- **TuSimple FPR**: False Positive Rate on TuSimple dataset (lower is better).
- **TuSimple FNR**: False Negative Rate on TuSimple dataset (lower is better).
- *Results reported on Test set with respective backbones.*

## Citation

If you use RowDetr or the dataset in your research, please cite:

```bibtex
@article{CHEPPALLY2025101494,
  title = {RowDetr: End-to-End Crop Row Detection Using Polynomials},
  author = {Rahul Harsha Cheppally and Ajay Sharda},
  journal = {Smart Agricultural Technology},
  pages = {101494},
  year = {2025},
  issn = {2772-3755},
  doi = {10.1016/j.atech.2025.101494},
  url = {https://www.sciencedirect.com/science/article/pii/S2772375525007257},
  keywords = {Crop row detection, Autonomous navigation, Agricultural Robotics, Attention mechanism}
}
```



## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## Acknowledgments

- Thanks to the [DETR](https://github.com/facebookresearch/detr) team for the foundational architecture.
- The dataset collection was supported by [relevant funding/institution, if applicable].
- Special thanks to contributors and early testers.

## Contact

For questions or issues:
- Open an [issue](https://github.com/r4hul77/RowDetr-v2/issues) on GitHub.
---

*Last updated: October 08, 2025 at 10:59 AM CDT*
