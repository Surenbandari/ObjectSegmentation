# Object Segmentation on ARMBench

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project for object detection and instance segmentation on the ARMBench dataset using Mask R-CNN with PyTorch.

## Overview

This implements object detection and instance segmentation for robotic perception tasks using the ARMBench dataset. The implementation uses Mask R-CNN with a ResNet50 backbone and Feature Pyramid Network (FPN), with custom architectural modifications to enhance segmentation performance.

### 🎯 Key Features

- **Instance Segmentation**: Detects and segments objects (totes and objects) in robotic manipulation scenarios
- **Multiple Test Scenarios**: Evaluates model performance on:
  - Mix-object-tote dataset
  - Same-object-transfer set
  - Zoomed-out-tote-transfer set
- **Two Training Configurations**: 
  - Small dataset (100 training images, 30 test images)
  - Large dataset (1000 training images, 300 test images)
- **Model Improvements**: Enhanced Mask R-CNN predictor with additional convolutional layers and ReLU activations

## Project Structure

```
Object-Segmentation-on-ARMBENCH/
├── README.md                                           # This file
├── requirements.txt                                    # Python dependencies
├── Object Segmentation on ARMBench.pptx               # Project presentation
│
├── notebooks/                                          # Data preprocessing notebooks
│   ├── ARMBENCH_json_file_conversions_100.ipynb       # Prepare 100-image dataset
│   └── ARMBENCH_json_file_conversions_1000.ipynb      # Prepare 1000-image dataset
│
├── scripts/                                            # Training & evaluation scripts
│   ├── object_detection_and_segmentation_on_armbench_100.py             # Baseline (100 images)
│   ├── object_detection_and_segmentation_on_armbench_100_improvement.py # Improved (100 images)
│   ├── object_detection_and_segmentation_on_armbench_1000.py            # Baseline (1000 images)
│   └── object_detection_and_segmentation_on_armbench_1000_improvement.py # Improved (1000 images)
│
└── visualization/                                      # Visualization scripts
    └── armbench_object_detection_and_segmentation_visulaization.py
```

## Dataset

### ARMBench Segmentation Dataset

The project uses the ARMBench Segmentation Dataset v0.1, which contains images of robotic manipulation scenarios with COCO-format annotations.

**Download Dataset:**
```bash
wget https://armbench-dataset.s3.amazonaws.com/segmentation/armbench-segmentation-0.1.tar.gz
tar -xzf armbench-segmentation-0.1.tar.gz
```

**Dataset Structure:**
- `mix-object-tote/`: Main training and testing images
- `same-object-transfer-set/`: Transfer learning test set
- `zoomed-out-tote-transfer-set/`: Zoomed-out test scenarios

**Dataset Splits:**
- **100-image configuration**: 100 train, 30 test (per test set)
- **1000-image configuration**: 1000 train, 300 test (per test set)

### Data Preprocessing

The Jupyter notebooks in the `notebooks/` directory handle:
1. Extracting subsets of images from the full dataset
2. Creating corresponding COCO annotation JSON files
3. Generating Excel files with image lists
4. Copying selected images to organized folders

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended)
- CUDA Toolkit and cuDNN (for GPU acceleration)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd Object-Segmentation-on-ARMBENCH
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the ARMBench dataset (see Dataset section above)

## Model Architecture

### Baseline Model
- **Base Architecture**: Mask R-CNN with ResNet50-FPN backbone
- **Pretrained Weights**: COCO pretrained
- **Classes**: 3 (Background, Tote, Object)

### Improved Model
Enhanced Mask R-CNN with modified mask predictor:
- Additional convolutional layer for better feature representation
- ReLU activations between conv layers
- Improved mask prediction capability

```python
class ModifiedMaskRCNNPredictor(nn.Module):
    def __init__(self, in_channels, hidden_layer, out_channels):
        super(ModifiedMaskRCNNPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_layer, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_layer, hidden_layer, kernel_size=3, padding=1)  # Intermediate layer
        self.conv3 = nn.Conv2d(hidden_layer, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
```

## Training

### Training Example

For 100-image improvement model:
```bash
python scripts/object_detection_and_segmentation_on_armbench_100_improvement.py
```

For 1000-image improvement model:
```bash
python scripts/object_detection_and_segmentation_on_armbench_1000_improvement.py
```

**Note**: The scripts are originally designed for Google Colab. For local execution, modify paths accordingly.

## Evaluation

Models are evaluated using:
- **mAP (Mean Average Precision)**: Primary metric using COCO evaluation
- **IoU Thresholds**: Standard COCO metrics (@0.5, @0.75, @0.5:0.95)

Test sets:
1. Mix-object-tote test set
2. Same-object-transfer set
3. Zoomed-out-tote-transfer set

## Visualization

The visualization script in the `visualization/` directory provides:
- Annotated images with bounding boxes
- Colored instance masks
- Class labels on detected objects

```bash
python visualization/armbench_object_detection_and_segmentation_visulaization.py
```

## Usage Example

### For Inference

```python
import torch
from PIL import Image

# Load trained model
model = torch.load("model_100.pt")
model.eval()

# Perform segmentation
img_path = "path/to/your/image.jpg"
img, pred_classes, masks = instance_segmentation(img_path, model, rect_th=5, text_th=4)

# Display results
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
```

## 📊 Results

The project evaluates model performance on three test scenarios:
1. **Mix-tote-object test**: Standard test set
2. **Same-object-transfer**: Transfer learning on same objects
3. **Zoomed-out-tote**: Generalization to different viewing angles

Results are measured using mAP (mean Average Precision) at various IoU thresholds.

## Requirements

See `requirements.txt` for complete list of dependencies.

Key libraries:
- PyTorch & TorchVision
- pycocotools
- OpenCV
- NumPy
- Matplotlib
- Pillow

## Acknowledgments

- **ARMBench Dataset Creators**: For providing the comprehensive segmentation dataset
- **PyTorch Team**: For the excellent deep learning framework
- **TorchVision Team**: For pre-trained models and utilities
- **COCO Team**: For the standardized evaluation metrics and tools

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors
- **Surendhar Bandari**

## Contact
- [ARMBench Dataset](https://armbench-dataset.s3.amazonaws.com/segmentation/armbench-segmentation-0.1.tar.gz)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [COCO Evaluation Metrics](https://cocodataset.org/#detection-eval)