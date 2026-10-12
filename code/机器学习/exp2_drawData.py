#coding=utf-8

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#读文本文件testSet.txt，然后使用散点图展示出来

data = []
label = []

f = open('exp2_SampleData.txt',"r") #-0.017612	14.053064	0
for line in f.readlines():
    line = line.strip().split()
    data.append([float(line[0]), float(line[1])])
    label.append(int(line[2]))
f.close()


data = array(data)
print(shape(data))
n = shape(data)[0] #number of points to create
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []
for i in range(n):
    if int(label[i])== 1:
        xcord1.append(data[i,0]); ycord1.append(data[i,1]) #画类别为1的样本散点
    else: #=0
        xcord2.append(data[i,0]); ycord2.append(data[i,1])#画类别为0的样本散点

fig = plt.figure()
ax = fig.add_subplot(111)
type1 = ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
type2 = ax.scatter(xcord2, ycord2, s=30, c='green')
x = arange(-3.0, 3.0, 0.1)

plt.xlabel('X1')
plt.ylabel('X2')
plt.show()