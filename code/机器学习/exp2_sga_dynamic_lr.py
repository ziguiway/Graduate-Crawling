#encoding=utf-8
#④ 2种以上动态学习率的随机梯度上升对比
#  方法1：反比衰减   alpha = 4/(1+j+i) + 0.0001   （教材 stocGradAscent1 的调度）
#  方法2：指数衰减   alpha = alpha0 * gamma^j     （每个epoch衰减一次）
#  方法3：阶梯衰减   每 stepEpoch 轮 alpha 减半

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

def sga_dynamic(dataMatrix, classLabels, numIter, schedule):
    """随机梯度上升，schedule(j, i, m) 返回第 j 轮第 i 次更新时的学习率"""
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    idx = np.arange(m)
    for j in range(numIter):
        np.random.shuffle(idx)                 #每轮随机打乱样本顺序
        for i in range(m):
            alpha = schedule(j, i, m)
            k = idx[i]
            h = sigmoid(sum(dataMatrix[k]*weights))
            error = classLabels[k] - h
            weights = weights + alpha * error * dataMatrix[k]
    return weights

def crossEntropy(dataMatrix, classLabels, weights):
    eps = 1e-12
    p = np.clip(sigmoid(dataMatrix @ weights), eps, 1-eps)
    return float(-np.mean(classLabels*np.log(p) + (1-classLabels)*np.log(1-p)))

if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    numIter = 50  #遍历数据集50遍

    schedules = {
        '反比衰减 4/(1+j+i)+1e-4': lambda j, i, m: 4/(1.0+j+i) + 0.0001,
        '指数衰减 0.1*0.9^j':      lambda j, i, m: 0.1 * 0.9**j,
        '阶梯衰减 每20轮减半':     lambda j, i, m: 0.1 * 0.5**(j//20),
    }

    results = {}
    for name, sch in schedules.items():
        np.random.seed(0)
        w = sga_dynamic(dataArr, labelMat, numIter, sch)
        loss = crossEntropy(dataArr, labelMat, w)
        acc = np.mean((sigmoid(dataArr @ w) > 0.5).astype(int) == labelMat)
        results[name] = (w, loss, acc)
        print(f'{name:<24} 权重 w = {np.round(w,4)}, 交叉熵损失 = {loss:.4f}, 训练准确率 = {acc:.2f}')

    #画图：散点 + 三条分隔线
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    pos = labelMat == 1
    ax.scatter(dataArr[pos,1], dataArr[pos,2], s=30, c='red', marker='s', label='class 1')
    ax.scatter(dataArr[~pos,1], dataArr[~pos,2], s=30, c='green', label='class 0')

    x = np.arange(-3.0, 3.0, 0.1)
    colors = ['purple', 'blue', 'orange']
    for c, (name, (w, _, _)) in zip(colors, results.items()):
        y = (-w[0] - w[1]*x) / w[2]
        ax.plot(x, y, c=c, label=name)
    ax.set_ylim(-6, 15)

    ax.set_xlabel('X1'); ax.set_ylabel('X2')
    ax.set_title('SGA with Dynamic Learning Rates (50 epochs)')
    ax.legend()
    plt.savefig('exp2_sga_dynamic_lr.png', dpi=150, bbox_inches='tight')
    print('已保存图像 exp2_sga_dynamic_lr.png')