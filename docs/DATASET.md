# Crop Row Detection Dataset

## Overview
This dataset is designed to support the development of vision-based strategies for autonomous robots navigating in GPS-denied agricultural environments. Crop row detection enables precise navigation for robots in challenging under-canopy environments. However, traditional vision-based methods often face difficulties due to gaps in crop rows, curved row patterns, and occlusions, which complicate accurate labeling and require extensive post-processing. To address these limitations, this dataset was meticulously collected and labeled to provide a robust resource for training and evaluating crop row detection algorithms.

## Dataset Structure
The dataset is split into three subsets: **Train**, **Test**, and **Validation**. Each subset contains:
- An **images** folder with image files capturing crop rows in under-canopy environments.
- A **labels** folder with corresponding JSON files for each image, where each JSON file is named `{image_name}.json` to match its respective image.

### JSON File Format
Each JSON file contains annotations for crop rows in the corresponding image. The structure of a JSON file is as follows:

- **`img_id`**: A unique identifier for the image (integer).
- **`labels`**: A list of objects, each representing a crop row in the image. Each object contains:
  - **`name`**: A string identifier for the crop row (e.g., `row_0`, `row_1`).
  - **`x`**: A list of x-coordinates (floats) representing points along the crop row.
  - **`y`**: A list of y-coordinates (floats) corresponding to the x-coordinates, defining the row's path.
  - **`alpha`**: A list of normalized distance values (floats between 0 and 1) representing the polynomial parameterization of the row, as described in [RowDetr: End-to-End Crop Row Detection Using Polynomials](https://arxiv.org/abs/2412.10525).

#### Example JSON
Below is an example JSON file illustrating the annotation format:

```json
{
  "img_id": 0,
  "labels": [
    {
      "name": "row_0",
      "x": [530.843, 588.246, ...],
      "y": [1189.066, 1044.191, ...],
      "alpha": [0.0, 0.15029798560170535, ...]
    },
    {
      "name": "row_1",
      "x": [1796.446, 1667.973, ...],
      "y": [830.979, 732.574, ...],
      "alpha": [0.0, 0.19830774044886185, ...]
    }
  ]
}
```

This example shows two crop rows (`row_0` and `row_1`) with their respective x, y coordinates and alpha values, where `alpha` represents the normalized distance along the row as per the polynomial-based approach outlined in the referenced paper.

## Usage
This dataset is particularly useful for researchers and developers working on:
- Autonomous navigation for agricultural robots.
- Vision-based crop row detection algorithms.
- Handling occlusions and curved crop rows in under-canopy environments.
- Developing end-to-end detection models, such as those using polynomial-based approaches.

The dataset is intended to reduce the dependency on post-processing steps and improve the robustness of crop row detection in challenging agricultural settings.

For more information please see our **[github repo](https://github.com/r4hul77/RowDetr-v2/tree/main)**.

## Citation
If you find this dataset useful for your research, please cite the following paper:

```bibtex
@article{CHEPPALLY2025101494,
  title = {RowDetr: End-to-End Crop Row Detection Using Polynomials},
  author = {Rahul Harsha Cheppally and Ajay Sharda},
  journal = {Smart Agricultural Technology},
  volume = {},
  number = {},
  pages = {101494},
  year = {2025},
  issn = {2772-3755},
  doi = {https://doi.org/10.1016/j.atech.2025.101494},
  url = {https://www.sciencedirect.com/science/article/pii/S2772375525007257},
  keywords = {Crop row detection, Autonomous navigation, Agricultural Robotics, Attention mechanism}
}
```

## Acknowledgments
This dataset was collected and labeled to overcome the challenges of vision-based crop row detection in GPS-denied, under-canopy environments. We hope it serves as a valuable resource for advancing agricultural robotics and autonomous navigation.