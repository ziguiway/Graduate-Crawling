# Q&A - HW2：分类（Phoneme Classification + Hessian Matrix）

李宏毅 HW2 作业遇到的问题和解答。

---

## 任务概览

HW2 分两个部分：

| 部分 | 主题 | 提交平台 |
|------|------|---------|
| HW2-1 | 音素分类（Phoneme Classification） | Kaggle |
| HW2-2 | Hessian Matrix（判断模型在局部最小值还是鞍点） | NTU COOL |

---

## HW2-1：Phoneme Classification

### 任务是什么

**多分类任务**：给定一帧语音的 429 维特征，预测它属于 39 个音素中的哪一个。

**啥是音素？**
- 跟音色/音量不是一回事
- 音色/音量是声音的物理属性（好不好听、大不大声）
- 音素是**语言里最小的发音单位**
- 比如你说「波」就是 `/p/` + `/o/` 两个音素，「妈」是 `/m/` + `/a/`

**类比理解**：语音就像一段连续的长录音。模型把它切成很多小片段（每段 25 毫秒，叫一帧），然后逐帧识别当时嘴巴在发哪个音。拼起来就是「第 0.00~0.03 秒发 `/s/`，第 0.03~0.06 秒发 `/p/`，第 0.06~0.09 秒发 `/i/`…」＝ `"spi..."`。

数据来自 TIMIT 语音库，本质上是**语音识别的前置任务**。

### 数据集

**.npy 是啥？** NumPy 的二进制文件格式。和 CSV 的区别：
- CSV：文本文件，能用记事本打开，但读写慢、占空间
- `.npy`：二进制文件，不能直接打开看，但**体积小、读写快**
- 加载：`data = np.load('xxx.npy')` → 直接得到 numpy 数组

> 不是女朋友（笑死），是 NumPy 数据。

| 文件                   | 形状               | 含义                  |
| -------------------- | ---------------- | ------------------- |
| `train_11.npy`       | (1,229,932, 429) | 122 万帧训练数据，每帧 429 维 |
| `train_label_11.npy` | (1,229,932,)     | 标签 0~38（共 39 个音素）   |
| `test_11.npy`        | (451,552, 429)   | 45 万帧测试数据           |

122 万帧什么概念？HW1 只有 2k 行数据，数据量大了 600 倍。

### 和 HW1 的核心区别

|               | HW1（回归）                  | HW2（分类）                      |
| ------------- | ------------------------ | ---------------------------- |
| 输出层           | `Linear(..., 1)` → 1 个数字 | `Linear(..., 39)` → 39 类概率   |
| 损失函数          | `MSELoss`                | **`CrossEntropyLoss`**       |
| 评价指标          | MSE / RMSE / R²          | **Accuracy（准确率）**            |
| 输出 activation | 无                        | CrossEntropyLoss 内部做 softmax |
| 数据量           | 2k 行                     | 122 万帧                       |

### 模型结构（baseline）

```
输入 (429)
  ↓
Linear(429 → 1024) + Sigmoid
  ↓
Linear(1024 → 512) + Sigmoid
  ↓
Linear(512 → 128) + Sigmoid
  ↓
Linear(128 → 39)
  ↓
输出 (39 类 logits) → CrossEntropyLoss
```

**关键细节：**
- 输出层 39 个节点，不加 activation（CrossEntropyLoss 内部做 softmax）
- 全程用 Sigmoid 做激活函数（不是 ReLU）
- Label 用 `LongTensor`（分类标签要求整数类型）

### CrossEntropyLoss 怎么用

```python
criterion = nn.CrossEntropyLoss()

# 模型输出：(batch_size, 39) — logits，不经过 softmax
outputs = model(inputs)

# 标签：(batch_size,) — 整数，每个值 0~38
_, train_pred = torch.max(outputs, 1)  # 取概率最大的类
loss = criterion(outputs, labels)      # 内部做 softmax + cross entropy
```

**跟 MSE 的区别：**
- MSE：预测值和真实值都是连续数，算差的平方
- CrossEntropy：模型吐出 39 个分数（logits），内部转成概率，跟 one-hot 标签比交叉熵

### 性能指标：Accuracy

HW1 看 loss 和 R²，HW2 看 accuracy：

```python
_, pred = torch.max(outputs, 1)                     # 取概率最大的类
acc = (pred.cpu() == labels.cpu()).sum().item() / len(labels)
```

baseline 模型跑 20 个 epoch 大概能到：
- Train Acc: ~79%
- Val Acc: ~70%

### 注意事项

**1. 数据量巨大，注意内存**
- 122 万帧加载进内存就要约 2GB
- 训练完后及时 `del` 大变量 + `gc.collect()` 释放内存

**2. 测试集没有 label**
```python
class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            self.label = torch.LongTensor(y)  # 训练集有标签
        else:
            self.label = None                  # 测试集无标签
```

**3. 提交格式**
```
Id,Class
0,23
1,15
2,8
...
```

---

## HW2-2：Hessian Matrix

### 任务是什么

训练完神经网络后，**判断模型停在什么位置**：

- **Local Minima**（局部最小值）— 梯度 ≈ 0，Hessian 所有特征值 > 0
- **Saddle Point**（鞍点）— 梯度 ≈ 0，Hessian 有正有负
- **None of the above** — 梯度还不为 0

### 判断规则

| 条件 | 结论 |
|------|------|
| 梯度范数 < 1e-3 **且** 正特征值比例 > 0.5 | Local Minima |
| 梯度范数 < 1e-3 **且** 正特征值比例 ≤ 0.5 | Saddle Point |
| 梯度范数 ≥ 1e-3 | None of the above |

### 怎么算的（不考推导）

- **梯度范数**：各层 weight 梯度的 L2 范数的平均值
- **Hessian 矩阵**：用 autograd-lib 库算，每个层算一个 Hessian → 求特征值 → 看正特征值占比
- 近似用的是 Gauss-Newton 法（忽略二阶项，只保留一阶项的外积）

### 实际操作

```python
student_id = '你的学号'  # 不同学生拿到的 checkpoint 不同
```

跑代码会输出：
```
gradient norm: 0.072, minimum ratio: 0.465
```

根据规则：gradient norm = 0.072 > 1e-3 → **None of the above**

---

## HW1 vs HW2 学到的新东西

| 概念   | HW1         | HW2              |
| ---- | ----------- | ---------------- |
| 问题类型 | 回归          | 分类               |
| 损失函数 | MSELoss     | CrossEntropyLoss |
| 评价指标 | RMSE / R²   | Accuracy         |
| 输出层  | 1 个节点       | C 个节点（C=类别数）     |
| 标签类型 | FloatTensor | LongTensor       |
| 数据量  | 2k          | 122 万            |
| 激活函数 | ReLU        | Sigmoid          |
