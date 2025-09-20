from ultralytics import YOLO

# Load a model
model = YOLO("model/GAI-YOLOv8.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model 不使用预训练权重，就注释这一行即可
# train
model.train(data='data/Chinese sturgeon.yaml',
            cache=False,
            imgsz=640,
            epochs=200,
            batch=4,
            close_mosaic=0,
            workers=0,
            device='0',
            optimizer='SGD',  # using SGD
            # amp=False,  # close amp
            project='runs/train',
            name='exp',
            patience=0,
            )

