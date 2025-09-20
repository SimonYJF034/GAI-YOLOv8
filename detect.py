from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'GAI-YOLOv8.pt') #model path or triton URL
    model.predict(source=r'dataset/images/test', #images path
                  imasz=640,
                  project='runs/detect',
                  name='exp'
                  save=True,
                  show=False,
                  )
