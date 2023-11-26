
import io
import numpy as np
import json
from PIL import Image
import requests
import threading
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

model = load_model('../train-project/cnn.model.resnet101.h5')
# pred的输入应该是一个images的数组，而且图片都已经转为numpy数组的形式
# pred = model.predict(['./validation/button/button-demoplus-20200216-16615.png'])

#这个顺序一定要与label.json顺序相同，模型输出是一个数组，取最大值索引为预测值
Label = [
    "button",
    "keyboard",
    "searchbar",
    "switch"
    ]

testPaths = [
    "./test/button.png",
    "./test/button2.jpg",
    "./test/keybord.jpg",
    "./test/keybord2.jpeg",
    "./test/keybord3.jpg",
    "./test/searchbar.png",
    "./test/searchbar2.jpg",
    "./test/searchbar3.jpg",
    "./test/switch.jpg",
    "./test/switch2.jpg",
    "./test/switch3.jpg",
    "./test/switch4.jpg",
    ]

images = []
for testPath in testPaths:
    image = cv2.imread(testPath)
    # image = cv2.cvtColor(image,cv2.CLOLR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image,(255,255))
    images.append(image)

images = np.array(images, dtype="float") / 255.0
pred = model.predict(images)

for i in range(np.shape(pred)[0]):
    _max = np.argmax(pred[i])
    print(_max)
    print(testPaths[i],Label[_max])
print(pred)
