# Python Q&A

学习过程中遇到的问题汇总，按主题分章节。

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

这是 PyTorch Dataset `__getitem__` 里的常见写法，用来逐条取样本。

---

## PyTorch 数据处理与训练

### 完整的 COVID-19 预测训练流程

**1. 数据加载与预处理**
```python
import pandas as pd

train_data = pd.read_csv('covid.train.csv').values
# 分离特征和标签：去掉第一列(id)，最后一列是标签
x_train, y_train = train_data[:, 1:-1], train_data[:, -1]
```

**2. Dataset 类标准写法**
```python
from torch.utils.data import Dataset
import torch

class COVID19Dataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)  # 转成 FloatTensor
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]              # 预测时只有特征
        else:
            return self.x[idx], self.y[idx] # 训练时有特征+标签

    def __len__(self):
        return len(self.x)
```

**3. DataLoader 用法**
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# shuffle=True: 训练时打乱数据，提高泛化能力
# batch_size: 每次迭代处理多少个样本
```

**4. 模型定义标准写法**
```python
import torch.nn as nn

class COVID19Model(nn.Module):
    def __init__(self):
        super(COVID19Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(93, 64),   # 输入维度要匹配特征数
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),     # 输出维度：回归任务是 1
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # squeeze 去掉多余的维度
```

**5. 训练四步曲**
```python
optimizer.zero_grad()   # 1. 清零梯度（必须！）
loss.backward()         # 2. 反向传播，计算梯度
optimizer.step()        # 3. 更新参数
```

**6. 损失函数与优化器**
```python
criterion = nn.MSELoss()           # 回归任务：MSE
optimizer = optim.Adam(model.parameters(), lr=0.001)
```


### numpy 切片语法 `[:, 1:-1]` 是什么意思？

**逗号分隔维度**：
- 逗号左边：操作行
- 逗号右边：操作列

```python
train_data[:, 1:-1]  # 所有行，第1列到倒数第2列
train_data[:, -1]    # 所有行，最后一列
```

**例子**：
```python
# 假设 train_data 是 3行 × 5列
# [ [1, 10, 20, 30, 100],
#   [2, 11, 21, 31, 101],
#   [3, 12, 22, 32, 102] ]

train_data[:, 1:-1]  # 结果: [[10,20,30], [11,21,31], [12,22,32]]
train_data[:, -1]    # 结果: [100, 101, 102]
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
# 例如 batch_size=16，输出形状是 (16, 1)

x = self.net(x)        # 形状: (16, 1)
x = x.squeeze(1)       # 形状: (16)  ← 移除第1维（长度为1的那个）

# 为什么需要这样做？
# 因为回归任务的标签 y 是一维的 (16)
# 如果输出是 (16, 1)，和标签形状不匹配，计算 loss 会报错
# squeeze 后变成 (16)，就能正常计算 MSELoss
```

**图解：**

```
原始输出:           squeeze(1) 后:
[16, 1]             [16]
┌─────────┐         ┌───┐
│ [0.5]   │         │0.5│
│ [0.3]   │   →     │0.3│
│ [0.8]   │         │0.8│
│  ...    │         │...│
└─────────┘         └───┘
```

**squeeze vs unsqueeze**

| 操作 | 作用 | 示例 |
|---|---|---|
| `squeeze(dim)` | 移除指定维度（长度必须为1） | `(16, 1)` → `(16)` |
| `unsqueeze(dim)` | 在指定位置添加新维度（长度为1） | `(16)` → `(16, 1)` |

**常见用法：**

```python
# 移除所有长度为1的维度
x = x.squeeze()       # (1, 16, 1) → (16)

# 只移除特定维度
x = x.squeeze(0)      # (1, 16, 1) → (16, 1)  第0维被移除
x = x.squeeze(1)      # (16, 1, 5) → (16, 5)  第1维被移除

# 如果指定维度长度不是1，squeeze 不做任何事
x = torch.randn(16, 5)
x = x.squeeze(1)      # 还是 (16, 5)，因为第1维长度是5，不是1
```

**在 COVID-19 预测任务中的应用：**

```python
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # 最后输出维度是 1
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (batch, 1) → (batch)
```

这样输出的形状就和标签 `y`（形状 `(batch)`）匹配了，可以直接计算 `MSELoss(pred, target)`。

---

### 为什么必须去掉那一维？不去会怎样？

**直接演示：**

```python
import torch
import torch.nn as nn

# 模拟真实情况
batch_size = 16
pred_with_dim = torch.randn(16, 1)   # 模型输出：(16, 1)
pred_squeezed = pred_with_dim.squeeze(1)  # squeeze后：(16)
target = torch.randn(16)             # 标签：(16)

criterion = nn.MSELoss()

# ✅ squeeze 后可以正常计算
loss = criterion(pred_squeezed, target)
print(loss)  # 正常输出一个数值

# ❌ 不 squeeze 会怎样？
# PyTorch 会广播 (16, 1) 和 (16)，结果变成 (16, 16)！
loss_wrong = criterion(pred_with_dim, target)
print(loss_wrong)  # 数值完全错误！
```

**为什么会出错？**

```
pred:     (16, 1)    ← 每个样本一个预测值，但包了一层
target:   (16)       ← 每个样本一个真实值

MSELoss 内部会做减法：
pred - target → (16, 1) - (16)

PyTorch 广播规则：把 (16) 当成 (1, 16)
结果变成：(16, 1) - (1, 16) = (16, 16)  ← 完全错了！

本来应该：16个样本，每个算一个误差
结果变成：16×16 = 256个误差值（每个预测和每个标签都算了一遍）
```

**图解广播灾难：**

```
pred (16, 1):          target (16) 广播成 (1, 16):
┌────┐                 ┌─────────────────────┐
│p1  │                 │t1  t2  t3  ...  t16 │
│p2  │                 └─────────────────────┘
│... │                 
│p16 │                 
└────┘                 

相减后变成 (16, 16):
┌──────────────────────────┐
│p1-t1  p1-t2  ...  p1-t16 │  ← 第1个预测和所有标签都算了误差！
│p2-t1  p2-t2  ...  p2-t16 │
│...                       │
│p16-t1 ...     p16-t16    │
└──────────────────────────┘
```

**总结：**

| 情况          | pred 形状   | target 形状 | 结果                       |
| ----------- | --------- | --------- | ------------------------ |
| ✅ 正确        | `(16)`    | `(16)`    | 16个误差，求平均，正确             |
| ❌ 不 squeeze | `(16, 1)` | `(16)`    | 广播成 (16, 16)，256个误差，完全错误 |
|             |           |           |                          |

所以 `squeeze(1)` 不是"可选的优化"，而是**必须做**，否则 loss 计算完全错误！

**本质：只要 pred 和 target 形状匹配就行**

```python
# 方式一：squeeze pred
pred = model(x).squeeze(1)   # (16, 1) → (16)
loss = criterion(pred, target)  # (16) vs (16) ✅

# 方式二：unsqueeze target
pred = model(x)              # (16, 1)
target = target.unsqueeze(1) # (16) → (16, 1)
loss = criterion(pred, target)  # (16, 1) vs (16, 1) ✅

# 两种方式都能正确计算，只要形状对得上！
```

`squeeze` 只是让形状匹配的手段之一，不是唯一解法。核心原则：**pred 和 target 的形状必须一致**。

---

## nn.Module 自定义网络

### 继承 nn.Module 必须实现什么？

**必须实现：**
- `__init__(self)` - 初始化，定义网络层
- `forward(self, x)` - 前向传播，定义数据流向

**模板：**

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # 必须调用父类初始化！
        # 定义网络层
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # 定义前向传播
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# 使用
model = MyModel()
output = model(input_data)  # 自动调用 forward
```

**关键点：**

| 方法 | 作用 | 注意事项 |
|---|---|---|
| `__init__` | 定义层结构 | 必须先 `super().__init__()` |
| `forward` | 定义计算流程 | 不要手动调用，用 `model(x)` 自动触发 |

**为什么不用手动调用 forward？**

```python
model = MyModel()

# ❌ 错误写法
output = model.forward(x)  # 不推荐

# ✅ 正确写法
output = model(x)          # 自动调用 forward，还会处理 hooks 等内部逻辑
```

`model(x)` 内部会调用 `forward`，但还会执行额外的钩子函数和内部处理。

**nn.Module 自动提供的功能：**

继承 `nn.Module` 后，自动获得：

```python
model = MyModel()

# 1. 参数管理
model.parameters()      # 获取所有可训练参数
model.named_parameters() # 获取参数名+参数

# 2. 设备转移
model.to('cuda')        # 移动到 GPU
model.to('cpu')         # 移动到 CPU

# 3. 模式切换
model.train()           # 训练模式（启用 Dropout、BatchNorm）
model.eval()            # 评估模式（禁用 Dropout、BatchNorm）

# 4. 模型保存/加载
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

这些都不用自己写，继承 `nn.Module` 就有了。

---

### 模型里的 `cal_loss` 函数是干嘛的？

**这不是 nn.Module 要求的方法，是自定义的辅助函数。**

```python
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(...)
        self.criterion = nn.MSELoss()  # 损失函数放在模型里

    def forward(self, x):
        return self.net(x).squeeze(1)

    # 自定义的辅助函数，不是必须的
    def cal_loss(self, pred, target):
        return self.criterion(pred, target)
```

**两种写法对比：**

```python
# 写法一：损失函数放模型里（HW1 的写法）
model = NeuralNet(input_dim)
pred = model(x)
loss = model.cal_loss(pred, target)  # 通过模型计算 loss

# 写法二：损失函数放外面（更常见的写法）
model = NeuralNet(input_dim)
criterion = nn.MSELoss()  # 损失函数独立定义
pred = model(x)
loss = criterion(pred, target)  # 直接调用
```

**为什么 HW1 把 loss 放模型里？**

因为注释里写了 `TODO: implement L2 regularization`，方便在 `cal_loss` 里加正则化：

```python
def cal_loss(self, pred, target):
    loss = self.criterion(pred, target)
    # L2 正则化（权重衰减）
    l2_reg = 0
    for param in self.parameters():
        l2_reg += param.norm(2)
    loss += 0.001 * l2_reg  # lambda = 0.001
    return loss
```

**总结：**

| 方法 | 是否必须 | 作用 |
|---|---|---|
| `__init__` | ✅ 必须 | 定义网络层 |
| `forward` | ✅ 必须 | 前向传播 |
| `cal_loss` | ❌ 自定义 | 封装损失计算，方便加正则化等 |

`cal_loss` 只是代码组织方式，不是 PyTorch 的要求。你也可以把 `criterion` 放外面，效果一样。

---

### 什么是 L2 正则化？

**正则化 = 防止过拟合**

过拟合：模型在训练集表现很好，测试集很差（学太"死"了）。

L2 正则化通过**惩罚大权重**，让模型更简单、更泛化。

**公式：**

```
原始损失:  Loss = MSE(pred, target)

L2 正则后: Loss = MSE(pred, target) + λ * Σ(w²)
                                    ↑
                              权重平方和
```

- `λ`：正则化系数，控制惩罚力度
- `Σ(w²)`：所有权重参数的平方和

**直观理解：**

```
没有 L2 正则：
权重可能很大 → 模型复杂 → 容易过拟合

有 L2 正则：
权重被"压"小 → 模型简单 → 泛化更好
```

**代码实现：**

```python
# 方式一：手动加 L2
def cal_loss(self, pred, target):
    loss = self.criterion(pred, target)
    l2_reg = 0
    for param in self.parameters():
        l2_reg += param.pow(2).sum()  # 权重平方和
    loss += 0.001 * l2_reg  # λ = 0.001
    return loss

# 方式二：用 optimizer 的 weight_decay（更常用）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
# weight_decay 就是 L2 正则的 λ
```

**方式二更简单，PyTorch 自动帮你加了 L2。**

**L1 vs L2 正则：**

| | L1 正则 | L2 正则 |
|---|---|---|
| 公式 | `+ λ * Σ|w|` | `+ λ * Σ(w²)` |
| 效果 | 让权重变成 **0**（稀疏） | 让权重变 **小**（但不为0） |
| 用途 | 特征选择 | 防过拟合 |

**图解：**

```
权重值变化：
无正则:  [2.5, -1.8, 0.9, 3.2, -0.5]  ← 数值较大
L2正则:  [1.2, -0.9, 0.4, 1.5, -0.2]  ← 数值被压小
L1正则:  [1.0, 0, 0, 1.2, 0]          ← 很多变成0
```

**总结：L2 正则 = 给 loss 加一个"权重平方和"的惩罚项，迫使权重变小，防止过拟合。**

