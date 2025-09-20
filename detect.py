from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'C:\Users\11566\Desktop\ultralytics-main\runs\train\exp32\weights\best.pt')
    model.predict(source=r'C:/Users/11566/Desktop/seg-data/images/test',
                  save=True,
                  show=False,
                  )
