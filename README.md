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
## Machine Learning Concepts Utilized in This Mask R-CNN Project

This project leverages several key machine learning and deep learning concepts that are integral to the Mask R-CNN architecture and training pipeline:

### 1. Convolutional Neural Networks (CNNs)
- Uses a deep CNN backbone (ResNet-50) with Feature Pyramid Networks (FPN) to extract rich multiscale feature representations from input images.
- CNNs automatically learn hierarchical features such as edges, textures, and object parts useful for detection and segmentation.

### 2. Region Proposal Network (RPN)
- Generates candidate object regions (bounding box proposals) from the backbone’s feature maps.
- Learns to propose regions likely to contain objects, reducing search space and improving efficiency versus sliding-window methods.

### 3. Two-Stage Detection Framework
- First stage: Propose regions via RPN.
- Second stage: Extract fixed-size features from proposed regions using RoIAlign, then:
  - Classify objects in each region.
  - Refine bounding box coordinates.
  - Predict segmentation mask for each object.

### 4. RoIAlign Layer
- Corrects spatial misalignments caused by quantization in previous RoI pooling.
- Uses bilinear interpolation to precisely extract fixed-size feature maps preserving spatial correspondence for each proposal.
- Improves mask quality especially for small objects.

### 5. Instance Segmentation via Mask Head
- A parallel branch to classification and bounding box regression predicts pixel-wise binary masks for each detected object.
- The mask head uses convolutional and upsampling layers to create high-resolution masks delineating object boundaries.

### 6. Multi-task Loss Optimization
- Combines multiple loss components during training:
  - Classification loss (cross-entropy) for detecting object classes.
  - Bounding box regression loss (smooth L1) for precise localization.
  - Mask segmentation loss (binary cross entropy) for mask accuracy.
- The network optimizes all these losses jointly for end-to-end learning.

### 7. Transfer Learning and Fine-tuning
- Starts with a model pretrained on COCO dataset.
- Replaces classification and mask heads to fit Pascal VOC classes.
- Fine-tunes entire network on Pascal VOC data for better specialization while leveraging learned features.

### 8. Data Augmentation & Dataset Splitting
- Uses train/val splits provided by VOC dataset.
- Uses data transformations (resizing, normalization) in data loaders.
- These augmentations improve generalization and robustness.

### 9. Batch Processing and GPU Acceleration
- Efficient batch loading with PyTorch `DataLoader`.
- Training and inference performed on GPU (if available) for performance.

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

---
## Workflow
### Data Preparation

- Download Pascal VOC 2012 dataset using kaggle API and unzip it.
- Two Dataset folders will be with structure:
  
VOC2012_Train_Val /Test <br />  
└── VOC2012_Train_Val/Test <br />  
├── JPEGImages/ # Raw image files (*.jpg) <br />  
│ ├── 000001.jpg <br />  
│ ├── 000002.jpg <br />  
│ └── ... <br />  
├── Annotations/ # XML files with bounding box and class annotations <br />  
│ ├── 000001.xml <br />  
│ ├── 000002.xml <br />  
│ └── ... <br />  
├── SegmentationClass/ # Semantic segmentation ground truths (PNG format) <br />  
│ ├── 000001.png <br />  
│ ├── 000002.png <br />  
│ └── ... <br />  
├── ImageSets/ <br />  
│ └── Main/ # Text files listing train/val/test splits <br />  
│ ├── train.txt <br />  
│ ├── val.txt <br />  
│ ├── trainval.txt <br />  
│ └── test.txt <br />  
└── SegmentationObject/ # Instance-wise segmentation masks (optional in some versions) <br />  

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

