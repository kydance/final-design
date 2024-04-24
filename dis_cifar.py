# -*- coding: utf-8 -*-
"""
@author: ispurs
cifar10 数据可视化
label:
      0 airplane
      1 automobile
      2 bird
      3 cat
      4 deer
      5 dog
      6 frog
      7 horse
      8 ship
      9 truck
"""

import numpy as np  
import matplotlib.pyplot as plt

filename = '/data_batch_1.bin'  # cifar10二进制文件路径

num = 10000 # 文中包含的图片数量

bytestream = open(filename, "rb")  
buf = bytestream.read(num * (1 + 32 * 32 * 3))  
bytestream.close()  

data = np.frombuffer(buf, dtype=np.uint8)  
data = data.reshape(num, 1 + 32*32*3)  
labels_images = np.hsplit(data, [1])  
labels = labels_images[0].reshape(num)  
images = labels_images[1].reshape(num, 32, 32, 3)  

numofimg = 0 # 图片序号

img = np.reshape(images[numofimg], (3, 32, 32)) #导出指定的图片
img = img.transpose(1, 2, 0)  

plt.figure(1)
plt.imshow(img)
plt.show()
print(labels[numofimg]) # 打印label信息