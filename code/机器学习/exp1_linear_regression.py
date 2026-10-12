#encoding=utf-8

#线性回归参数学习

import numpy as np
import matplotlib.pyplot as plt


#随机生成一组一维属性样本及其标签对 (xi , yi), i=1,2,…,m，m为样本的大小，这里为100
X= np.random.rand(100,1) # 随机生成100行1列数据X
Y= 4 + 3*X + np.random.randn(100,1) #得到Y
plt.scatter(X,Y) #绘制散点图
plt.show()

#这里采用梯度下降方法学习参数
X_b = np.c_[np.ones((100,1)),X] #把参数b吸纳到向量X中，得到向量X_b=[1,X]
eta=1.5 #初始化学习率
n_iterations=50 #迭代次数
m=100 #样本个数
theta=np.random.randn(2,1) #随机初始化参数b,w
plt.figure()
plt.ion()

for iteration in range(n_iterations):
    gradients = 1/m * X_b.T.dot(X_b.dot(theta)-Y) #计算梯度，注意为向量计算
    theta = theta - eta * gradients #更新参数
    #② 打印每一轮参数优化后的损失（均方误差 MSE = 1/m * Σ(ŷi-yi)²）
    loss = np.mean((X_b.dot(theta)-Y)**2)
    print(f'第{iteration+1}轮迭代后的损失MSE: {loss:.4f}')
    X_new=np.array([[0],[1]])
    X_new_b=np.c_[np.ones((2,1)),X_new]
    Y_predict=X_new_b.dot(theta)
    plt.cla()
    plt.plot(X, Y, "b.")  # 原始散点图
    plt.plot(X_new,Y_predict,"r-")
    plt.pause(0.2) #形成延时达到查看动画效果。



plt.axis([0,1,0,8])
plt.plot(X,Y,"b.") #原始散点图
X_new = np.array([[0], [1]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
Y_predict = X_new_b.dot(theta)
plt.plot(X_new, Y_predict, "b*")
plt.ioff()
plt.show()
print(f"学习到的最优参数为：w={theta[0]}和b={theta[1]}")

