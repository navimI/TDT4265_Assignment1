from ultralytics import YOLO

model = YOLO('yolov8n.yaml')  # build a new model from YAML
model.train(data='training/dataset.yaml', epochs=1, imgsz=512)
