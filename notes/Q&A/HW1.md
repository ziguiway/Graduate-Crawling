# Q&A - HW1：回归（COVID-19 预测）

做李宏毅 HW1 回归作业时遇到的问题和解答。

---

## Pandas

### iloc 是怎么取数据的？

`iloc` 基于整数位置（integer location）取数据，只认位置索引，不认列名或行标签。

**取行**
```python
df.iloc[0]       # 第0行，返回 Series
df.iloc[1]       # 第1行
df.iloc[-1]      # 最后一行
```

**取行+列**
```python
df.iloc[0, 1]    # 第0行第1列 → 具体值
df.iloc[1, 2]    # 第1行第2列
```

**切片**
```python
df.iloc[0:2]          # 前2行（左闭右开）
df.iloc[:, 1:3]       # 所有行，第1、2列
df.iloc[0:2, 1:3]     # 前2行，第1、2列
```

**花式索引**
```python
df.iloc[[0, 2]]       # 第0行和第2行
df.iloc[:, [0, 2]]    # 第0列和第2列
```

**iloc vs loc**

| | `iloc` | `loc` |
|---|---|---|
| 按什么取 | **位置**（整数） | **标签**（行名/列名） |
| 切片规则 | 左闭右开 `0:2` → 0,1 | 左闭右闭 `0:2` → 0,1,2 |

**在 Dataset 中的用法**
```python
self.data.iloc[index]  # 按整数位置取第 index 行，返回 Series
```

---

### numpy 切片语法 `[:, 1:-1]` 是什么意思？

**逗号分隔维度**：
- 逗号左边：操作行
- 逗号右边：操作列

```python
train_data[:, 1:-1]  # 所有行，第1列到倒数第2列
train_data[:, -1]    # 所有行，最后一列
```

**常用写法**：

| 写法 | 含义 |
|---|---|
| `:` | 所有 |
| `1:-1` | 从第1个到倒数第1个之前（不包含最后一个）|
| `-1` | 最后一个 |
| `:3` | 前3个（0,1,2） |
| `2:` | 从第2个开始到最后 |

---

## PyTorch 张量操作

### `.squeeze(1)` 是什么意思？

**squeeze** 的作用是**移除长度为1的维度**。

```python
# 假设模型输出形状是 (batch_size, 1)
x = self.net(x)        # 形状: (16, 1)
x = x.squeeze(1)       # 形状: (16)
```

**为什么需要？** 回归任务的标签 y 是一维的 (16)，如果输出是 (16, 1) 和标签形状不匹配。

**squeeze vs unsqueeze**

| 操作 | 作用 | 示例 |
|---|---|---|
| `squeeze(dim)` | 移除指定维度（长度必须为1） | `(16, 1)` → `(16)` |
| `unsqueeze(dim)` | 在指定位置添加新维度 | `(16)` → `(16, 1)` |

### 为什么必须去掉那一维？不去会怎样？

**MSELoss 广播灾难：**

```python
pred:     (16, 1)    ← 每个样本一个预测值，但包了一层
target:   (16)       ← 每个样本一个真实值

MSELoss 内部做减法：pred - target → (16, 1) - (16)
PyTorch 广播规则：把 (16) 当成 (1, 16)
结果变成：(16, 1) - (1, 16) = (16, 16)  ← 完全错了！
```

**核心原则：pred 和 target 的形状必须一致**

```python
# 方式一：squeeze pred
pred = model(x).squeeze(1)   # (16, 1) → (16)
loss = criterion(pred, target)  # ✅

# 方式二：unsqueeze target
pred = model(x)              # (16, 1)
target = target.unsqueeze(1) # (16) → (16, 1)
loss = criterion(pred, target)  # ✅
```

### `torch.cat(preds, dim=0).numpy()` 是什么意思？

测试阶段把多个 batch 的预测结果合并成一个大数组的标准写法。

```python
preds = []
for x in tt_set:
    pred = model(x)
    preds.append(pred)

# cat: 把 list 里的小 tensor 按第0维拼起来
torch.cat(preds, dim=0)  # shape: (总样本数,)

# .numpy(): 转成 numpy 数组
```

**核心逻辑：沿着指定的维度拼接**

```python
torch.cat([tensor1, tensor2, ...], dim=要拼接的维度)
```

**规则：** 除了 `dim` 指定的维度，其他维度大小必须完全相同。

**图解：**
```
dim=0 拼接:   [a]      →    [a]
              [b]           [b]   ← 往下接，行数变多

dim=1 拼接:   [a][b]   →    [a b]  ← 往右接，列数变多
```

---

## PyTorch 模型定义

### 继承 nn.Module 必须实现什么？

**必须实现：**
- `__init__(self)` - 定义网络层
- `forward(self, x)` - 前向传播

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

**自动获得的功能：**
```python
model.parameters()           # 所有可训练参数
model.to('cuda')             # 移到 GPU
model.train() / model.eval() # 模式切换
torch.save(model.state_dict(), 'model.pth')
```

### 模型里的 `cal_loss` 函数是干嘛的？

**不是 nn.Module 要求的方法，是自定义辅助函数。** HW1 把 loss 放模型里，方便以后加 L2 正则化：

```python
def cal_loss(self, pred, target):
    loss = self.criterion(pred, target)
    l2_reg = 0
    for param in self.parameters():
        l2_reg += param.norm(2)
    loss += 0.001 * l2_reg
    return loss
```

两种写法效果一样：
- HW1 写法：loss 放模型里，通过 `model.cal_loss(pred, target)` 调用
- 常见写法：loss 放外面，通过 `criterion(pred, target)` 调用

---

### 什么是 L2 正则化？

**防止过拟合**，通过惩罚大权重让模型更简单。

```
原始 Loss = MSE(pred, target)
L2 正则后 Loss = MSE(pred, target) + λ * Σ(w²)
```

**代码实现：**
```python
# 方式一：手动加 L2
def cal_loss(self, pred, target):
    loss = self.criterion(pred, target)
    l2_reg = sum(param.pow(2).sum() for param in self.parameters())
    return loss + 0.001 * l2_reg

# 方式二：用 weight_decay（更常用）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
```

### Optimizer 和 Loss 的关系是什么？

**Loss 告诉模型"错在哪"，Optimizer 告诉模型"怎么改"。**

**训练四步曲：**
```python
optimizer.zero_grad()   # 1. 清零梯度（必须！）
loss.backward()         # 2. 反向传播，计算梯度
optimizer.step()        # 3. 更新参数
```

**为什么需要 zero_grad？** PyTorch 默认累积梯度，不清零梯度会叠加。

**常见 Optimizer：**

| Optimizer | 特点 |
|---|---|
| SGD | 最简单，稳定但慢 |
| Adam | 自适应学习率，快且稳，**最常用** |
| AdamW | Adam + L2 正则改进，适合 Transformer |

---

## 特征工程：归一化

### 为什么必须做特征归一化？

COVID-19 数据量级差异大：
- 州 one-hot：0 或 1
- 调查数据：十几、几十

不归一化：数值大的特征支配整个输出，其他特征被忽略。

### 归一化公式（Z-Score）

```python
mean = self.data[:, 40:].mean(dim=0, keepdim=True)
std = self.data[:, 40:].std(dim=0, keepdim=True)
self.data[:, 40:] = (self.data[:, 40:] - mean) / std
```

其他归一化方法：

| 方法 | 公式 | 适用场景 |
|---|---|---|
| Z-Score | `(x-μ)/σ` | 深度学习默认 |
| Min-Max | `(x-min)/(max-min)` | 缩放到 [0,1] |

---

## 常见错误

### `TypeError: object of type has no len()`

**原因：** Dataset 类没有实现 `__len__` 方法。

```python
class COVID19Dataset(Dataset):
    def __init__(self, ...):
        self.data = ...

    def __getitem__(self, index):
        ...

    # ✅ 必须加上！
    def __len__(self):
        return len(self.data)
```

### COVID-19 作业 loss 不对/收敛慢

1. **没有做特征归一化（最常见）**
2. **Adam 学习率太大**：推荐 0.001，不是 0.01
3. **early_stop 设置太大**：50 就够，不用 200

### 常见 PyTorch 错误

| 错误 | 原因 | 解决 |
|---|---|---|
| Tensor on CPU/GPU mismatch | 设备和模型不在同一设备 | `x = x.to(device)` |
| Size mismatch | 维度不匹配 | `transpose/squeeze/unsqueeze` |
| CUDA out of memory | GPU 内存不足 | 减小 batch_size |
| Long vs Float mismatch | 张量类型不匹配 | `labels = labels.long()` |

---

## Loss 函数怎么判断好不好

### MSE — 均方误差

**原理：** 预测值和真实值之差，平方一下再平均。为什么平方？一是消除正负，二是放大偏差——差 2 变成 4，差 5 变成 25，**大误差被狠狠惩罚**。缺点是单位也跟着平方了，不好直接说"偏了多少"。

### RMSE — 均方根误差

**原理：** 把 MSE 开根号，**单位变回和原始数据一样**。比如 RMSE = 1.1，意思就是"平均偏了 1.1 个单位"。

### R² — 决定系数

**原理：** 跟"永远猜平均值"比。如果你一直猜平均值 16.43，误差就是数据本身的方差（58.03）。你的模型误差（MSE=1.20）只有这个的 2%。**R² 不看数据范围，最公平。**

**你的结果：**

| 指标 | 值 | 怎么判断 |
|------|----|---------|
| MSE | 1.20 | 越小越好 |
| RMSE | **1.10** | 平均偏 1.1，数据范围 2~41 |
| R² | **0.98** | 比瞎猜好 98% ✅ |

**早停** = 连续 N 轮没进步就停，**不是 bug，是保护**。你设 2 太少了（一震荡就停），建议改 20~50。

**学习曲线：** train loss 和 dev loss 画一起。两条线接近 ✅，dev loss 反弹 → 过拟合 ⚠️。
