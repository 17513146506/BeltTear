BeltTear-seg

A lightweight and high-precision instance segmentation model for conveyor belt tear detection, based on YOLOv8n-seg.
This repository provides the trained weights, dataset samples, configuration files, and source code for reproducing the experiments in our paper:

BeltTear-seg: A Lightweight Model for Belt Tear Detection with Squeeze Attention and Efficient Classification

📂 Repository Structure
.
├── config/                # Configuration files (e.g., configs.yaml)
├── evaluation_results/    # Evaluation logs and outputs
├── images/                # Sample input images
├── results/               # Visualization results of experiments
├── ultralytics/           # Modified YOLOv8n-seg source code
├── weights/               # Pretrained and trained model weights
├── classification1.py     # Classification module
├── predict.py             # Inference script
├── test_detect_mask.py    # Testing and mask evaluation
├── train_instance.py      # Instance segmentation training
├── train_rgb.py           # RGB training script
├── train_rgb_ir_feature_enhance.py  # RGB+IR feature enhancement training
📌Dataset

The BeltTear dataset contains 4,050 industrial belt images (severe tears, minor tears, and non-tears).
We provide part of the dataset covering different scenes.

Format: BMP images (2048 × 2000)

Annotation: JSON files from LabelMe

📌 Training
python train_instance.py --cfg config/configs.yaml --data data.yaml --weights yolov8n-seg.pt

📌Inference
python predict.py --weights weights/belttear_best.pt --source images/test.jpg

📊 Results

Accuracy: 95.9%

Improvement: +5.8% over YOLOv8n-seg on our custom dataset

Inference Speed: 12.5 ms per frame (real-time capable)
