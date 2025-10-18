BeltTear-seg

A lightweight and high-precision instance segmentation model for conveyor belt tear detection, based on YOLOv8n-seg.
This repository provides the trained weights, dataset samples, configuration files, and source code for reproducing the experiments in our paper

📂 Repository Structure
├── config/                # Configuration files (e.g., configs.yaml)
├── images/                # Sample input images
├── results/               # Visualization results of experiments
├── ultralytics/           # Modified YOLOv8n-seg source code
📌Dataset

The BeltTear dataset contains 4,050 industrial belt images (severe tears, minor tears).
We provide part of the dataset covering different scenes.

Format: BMP images (2048 × 2000)

Annotation: JSON files from LabelMe

📌 Training
python train_instance.py --cfg config/configs.yaml --data data.yaml --weights yolov8n-seg.pt

📊 Results

Accuracy: 95.9%

Improvement: +5.8% over YOLOv8n-seg on our custom dataset

Inference Speed: 12.5 ms per frame (real-time capable)
