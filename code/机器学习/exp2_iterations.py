#encoding=utf-8
#② 简单（批量）梯度上升迭代 5 次、50 次、100 次的对比实验

import matplotlib
matplotlib.use('Agg')  # 无界面环境直接出图；本地运行可注释掉此行以弹窗显示
import numpy as np
import matplotlib.pyplot as plt

#-------------- 与 exp2 相同的加载与模型部分 --------------
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

def gradAscent(dataMatrix, labelMat, maxCycles):
    """简单（批量）梯度上升，迭代 maxCycles 次"""
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    weights = np.ones((n,1))
    losses = []
    for k in range(maxCycles):
        h = sigmoid(dataMatrix @ weights)
        error = labelMat.reshape(-1,1) - h
        weights = weights + alpha * dataMatrix.T @ error
        # 交叉熵损失（越小越好，用于观察优化进度）
        eps = 1e-12
        p = np.clip(h, eps, 1-eps)
        loss = -np.mean(labelMat.reshape(-1,1)*np.log(p) + (1-labelMat.reshape(-1,1))*np.log(1-p))
        losses.append(float(loss))
    return weights, losses
#-------------- 对比 5 / 50 / 100 次迭代 --------------
if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()

    results = {}
    for n_iter in (5, 50, 100):
        w, losses = gradAscent(dataArr, labelMat, n_iter)
        results[n_iter] = (w, losses)
        print(f'迭代{n_iter:>3}次后的权重 w = {w.ravel()}, 交叉熵损失 = {losses[-1]:.4f}')

    # 画图：散点 + 三条分隔线
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    pos = labelMat == 1
    ax.scatter(dataArr[pos,1], dataArr[pos,2], s=30, c='red', marker='s', label='class 1')
    ax.scatter(dataArr[~pos,1], dataArr[~pos,2], s=30, c='green', label='class 0')

    x = np.arange(-3.0, 3.0, 0.1)
    colors = {5:'orange', 50:'purple', 100:'blue'}
    for n_iter, (w, _) in results.items():
        y = (-w[0,0] - w[1,0]*x) / w[2,0]
        ax.plot(x, y, c=colors[n_iter], label=f'{n_iter} iterations')

    ax.set_xlabel('X1'); ax.set_ylabel('X2')
    ax.set_title('Batch Gradient Ascent: 5 vs 50 vs 100 iterations')
    ax.legend()
    plt.savefig('exp2_iterations_compare.png', dpi=150, bbox_inches='tight')
    print('已保存图像 exp2_iterations_compare.png')
