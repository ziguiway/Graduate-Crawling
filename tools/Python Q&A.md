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

