from ultralytics import YOLO
import torch.multiprocessing as mp

# def train_yolov8():
model = YOLO("/home/air/projects/ultralytics-main/ultralytics/cfg/models/v8/yolov8-seg.yaml")  # build a new model from YAML
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("/home/air/projects/ultralytics-main/ultralytics/cfg/models/v8/yolov8-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/home/air/projects/ultralytics-main/ultralytics/cfg/datasets/seg.yaml", epochs=100, imgsz=1024)

