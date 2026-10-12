#encoding=utf-8
#③ 3个不同学习率下的简单随机梯度上升(stocGradAscent0)学习结果对比

import matplotlib
matplotlib.use('Agg')  # 无界面环境直接出图；本地运行可注释掉此行以弹窗显示
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('exp2_SampleData.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return np.array(dataMat), np.array(labelMat)

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def stocGradAscent0(dataMatrix, classLabels, alpha):
    """简单随机梯度上升：每次用一个样本的梯度更新权重，遍历一遍数据集"""
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)   #初始化为全1
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def crossEntropy(dataMatrix, classLabels, weights):
    eps = 1e-12
    p = np.clip(sigmoid(dataMatrix @ weights), eps, 1-eps)
    return float(-np.mean(classLabels*np.log(p) + (1-classLabels)*np.log(1-p)))

if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()

    alphas = [0.001, 0.01, 0.1]   #3个不同学习率
    results = {}
    for a in alphas:
        w = stocGradAscent0(dataArr, labelMat, a)
        loss = crossEntropy(dataArr, labelMat, w)
        acc = np.mean((sigmoid(dataArr @ w) > 0.5).astype(int) == labelMat)
        results[a] = (w, loss, acc)
        print(f'alpha={a:<6} 权重 w = {np.round(w,4)}, 交叉熵损失 = {loss:.4f}, 训练准确率 = {acc:.2f}')

    #画图：散点 + 三条分隔线
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    pos = labelMat == 1
    ax.scatter(dataArr[pos,1], dataArr[pos,2], s=30, c='red', marker='s', label='class 1')
    ax.scatter(dataArr[~pos,1], dataArr[~pos,2], s=30, c='green', label='class 0')

    x = np.arange(-3.0, 3.0, 0.1)
    colors = {0.001:'orange', 0.01:'purple', 0.1:'blue'}
    for a, (w, _, _) in results.items():
        y = (-w[0] - w[1]*x) / w[2]
        ax.plot(x, y, c=colors[a], label=f'alpha={a}')
    ax.set_ylim(-6, 15)

    ax.set_xlabel('X1'); ax.set_ylabel('X2')
    ax.set_title('Stochastic Gradient Ascent with Different Learning Rates')
    ax.legend()
    plt.savefig('exp2_sga_lr_compare.png', dpi=150, bbox_inches='tight')
    print('已保存图像 exp2_sga_lr_compare.png')