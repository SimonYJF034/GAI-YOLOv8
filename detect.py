# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'C:\Users\11566\Desktop\ultralytics-main\runs\train\exp32\weights\best.pt')
    model.predict(source=r'C:/Users/11566/Desktop/seg-data/images/test',
                  save=True,
                  show=False,
                  )