# Mask R-CNN for Object Detection and Instance Segmentation

## Overview

This project uses the Mask R-CNN architecture implemented in PyTorch to perform object detection and instance segmentation on the Pascal VOC 2012 dataset. Mask R-CNN extends Faster R-CNN by adding a parallel branch that predicts segmentation masks for each detected object.

---

## How Mask R-CNN Works

Mask R-CNN operates in three main stages:

1. **Backbone and Feature Pyramid Network (FPN):**
   Extracts feature maps from the input image at multiple scales using a ResNet-50 backbone with FPN.

2. **Region Proposal Network (RPN):**
   Proposes candidate object bounding boxes based on the anchor boxes in the feature maps.

3. **RoI Heads:**
   For each proposed region, the model:
   - Classifies the object and refines bounding boxes.
   - Predicts a binary mask for each class inside the bounding box.

This approach allows simultaneous object detection and pixel-level segmentation with high accuracy.

---

## File Structure

- `voc_dataset.py`: Dataset class for parsing Pascal VOC images and annotations.
- `train.py`: Training loop handling epochs, batch loading, and optimization.
- `evaluate.py`: Testing/inference loop to generate predictions on validation/test data.
- `visualize.py`: Tools for visualizing predicted bounding boxes, masks, and class labels over images.
- `utils.py`: Helper functions, e.g., transforms, collate functions.
- `README.md`: This document.

---
## Core Functions in the Code

### `get_model_instance_segmentation(num_classes)`

- Loads the pretrained Mask R-CNN model with COCO weights.
- Replaces the **box predictor** head with a new one to output logits for `num_classes` (21 for Pascal VOC including background).
- Replaces the **mask predictor** head similarly for the segmentation masks.
- This enables fine-tuning the model on the Pascal VOC dataset with the correct number of classes.

### `train_one_epoch(model, optimizer, data_loader, device, epoch)`

- Sets the model to training mode.
- Loops over batches of images and targets from the data loader.
- Moves data to the appropriate device (CPU/GPU).
- Performs forward pass, computes losses (box regression, classification, mask prediction).
- Performs backward pass and optimizer step to update weights per batch.
- Provides progress output per epoch.

### Training Loop in `main()`

- Iterates over a predefined number of epochs.
- Calls `train_one_epoch` to perform training on the entire dataset.
- Steps the learning rate scheduler.
- Prints epoch completion for monitoring.

### `evaluate_model(model, data_loader, device)`

- Switches the model to evaluation mode.
- Iterates over test/validation data without computing gradients.
- Gets predictions on input images (bounding boxes, masks, labels, scores).
- Returns the collected results for further analysis or visualization.

### `visualize_prediction(image, prediction, threshold=0.5)`

- Takes a single image and its model prediction output.
- Draws bounding boxes, class labels (using Pascal VOC class names), and confidence scores if above threshold.
- Overlays predicted binary masks with transparency.
- Uses Matplotlib for rich visual output.
- Helps qualitatively verify the model's performance.

---
## Dependencies

- Python 3.7+
- PyTorch 1.13+
- torchvision 0.14+
- PIL / Pillow
- numpy
- matplotlib
- **Kaggle API** for downloading the dataset programmatically

### Installing the Kaggle API

You can install the official Kaggle API Python package via pip:

---
## Workflow
### Data Preparation

- Download Pascal VOC 2012 dataset using kaggle API and unzip it.
- Two Dataset folders will be with structure:
VOC2012_Train_Val
└── VOC2012_Train_Val
├── JPEGImages/ # Raw image files (*.jpg)
│ ├── 000001.jpg
│ ├── 000002.jpg
│ └── ...
├── Annotations/ # XML files with bounding box and class annotations
│ ├── 000001.xml
│ ├── 000002.xml
│ └── ...
├── SegmentationClass/ # Semantic segmentation ground truths (PNG format)
│ ├── 000001.png
│ ├── 000002.png
│ └── ...
├── ImageSets/
│ └── Main/ # Text files listing train/val/test splits
│ ├── train.txt
│ ├── val.txt
│ ├── trainval.txt
│ └── test.txt
└── SegmentationObject/ # Instance-wise segmentation masks (optional in some versions)

VOC2012_Test
└── VOC2012_Test
├── JPEGImages/ # Raw image files (*.jpg)
│ ├── 000001.jpg
│ ├── 000002.jpg
│ └── ...
├── Annotations/ # XML files with bounding box and class annotations
│ ├── 000001.xml
│ ├── 000002.xml
│ └── ...
├── SegmentationClass/ # Semantic segmentation ground truths (PNG format)
│ ├── 000001.png
│ ├── 000002.png
│ └── ...
├── ImageSets/
│ └── Main/ # Text files listing train/val/test splits
│ ├── train.txt
│ ├── val.txt
│ ├── trainval.txt
│ └── test.txt
└── SegmentationObject/ # Instance-wise segmentation masks (optional in some versions)
- Update `root_folder` and file paths accordingly in code.

### Training

- Load model pretrained on COCO, replace heads for VOC classes.
- Train with batch size 1-2 for 20-50 epochs.
- Use provided training loop and optimizer setup.

Example training loop snippet:

- Update `root_folder` and file paths accordingly in code.

### Training

- Load model pretrained on COCO, replace heads for VOC classes.
- Train with batch size 1-2 for 20-50 epochs.
- Use provided training loop and optimizer setup.

### Evaluation

- Run model on validation/test data in evaluation mode.
- Generate predictions and visualize overlayed masks, bounding boxes, and labels.

---
## References

- [Torchvision Mask R-CNN](https://pytorch.org/vision/stable/models.html#mask-rcnn)
- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [PyTorch Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

---

## License

MIT License

