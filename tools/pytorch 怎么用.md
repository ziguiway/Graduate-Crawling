![](../code/Lhy_Machine_Learning/2022%20ML/00%20Colab&Pytorch/Pytorch_Tutorial_1.pdf)
![](../code/Lhy_Machine_Learning/2022%20ML/00%20Colab&Pytorch/Pytorch_Tutorial_2.pdf) 
## 一、PyTorch 简介

PyTorch 是一个 Python 机器学习框架，主要有两大特性：
- **N 维张量计算**：类似 NumPy，支持 GPU 加速
- **自动微分**：用于训练深度神经网络

---

## 二、核心概念

> 📊 **相关图表**：[PyTorch训练流程.excalidraw](PyTorch训练流程.excalidraw)

### 2.1 Tensor（张量）

张量是高维矩阵/数组，支持各种数学运算。

#### 创建张量
```python
import torch

# 从数据创建
x = torch.tensor([[1, -1], [-1, 1]])
x = torch.from_numpy(np.array([[1, -1], [-1, 1]]))

# 创建全零/全一张量
x = torch.zeros([2, 2])
x = torch.ones([1, 2, 5])
```

#### 常用操作
```python
# 算术运算
z = x + y
z = x - y
y = x.sum()
y = x.mean()
y = x.pow(2)

# 转置
x = x.transpose(0, 1)

# Squeeze/Unsqueeze
x = x.squeeze(0)   # 移除长度为1的维度
x = x.unsqueeze(1) # 扩展新维度

# 拼接
w = torch.cat([x, y, z], dim=1)
```

#### 设备管理
```python
# 检查 GPU 是否可用
torch.cuda.is_available()

# 移动到 CPU/GPU
x = x.to('cpu')
x = x.to('cuda')
x = x.to('cuda:0')  # 指定 GPU
```

#### 梯度计算
```python
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
z = x.pow(2).sum()
z.backward()
print(x.grad)  # 查看梯度
```

---

### 2.2 Dataset & Dataloader

> 📊 **相关图表**：[Dataset_DataLoader架构.excalidraw](Dataset_DataLoader架构.excalidraw)

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, file):
        # 读取数据 & 预处理
        self.data = ...

    def __getitem__(self, index):
        # 返回单个样本
        return self.data[index]

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

# 使用
dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

**注意**：
- 训练时设置 `shuffle=True`
- 测试时设置 `shuffle=False`

---

### 2.3 定义神经网络

> 📊 **相关图表**：[神经网络结构示例.excalidraw](神经网络结构示例.excalidraw)

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

#### 常用层
- `nn.Linear(in_features, out_features)` - 全连接层
- `nn.Sigmoid()` - Sigmoid 激活函数
- `nn.ReLU()` - ReLU 激活函数

---

### 2.4 损失函数

```python
import torch.nn as nn

# 回归任务
criterion = nn.MSELoss()

# 分类任务
criterion = nn.CrossEntropyLoss()

# 计算损失
loss = criterion(model_output, expected_value)
```

---

### 2.5 优化器

```python
import torch.optim

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0)

# 每个批次：
optimizer.zero_grad()   # 清零梯度
loss.backward()         # 反向传播
optimizer.step()        # 更新参数
```

---

## 三、完整训练流程

### 3.1 训练设置
```python
# 数据
dataset = MyDataset(file)
tr_set = DataLoader(dataset, batch_size=16, shuffle=True)

# 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyModel().to(device)

# 损失函数 & 优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

### 3.2 训练循环
```python
for epoch in range(n_epochs):
    model.train()  # 设置为训练模式
    for x, y in tr_set:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
```

### 3.3 验证循环
```python
model.eval()  # 设置为评估模式
total_loss = 0
for x, y in dv_set:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():  # 禁用梯度计算
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.cpu().item() * len(x)
avg_loss = total_loss / len(dv_set.dataset)
```

### 3.4 测试循环
```python
model.eval()
preds = []
for x in tt_set:
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
        preds.append(pred.cpu())
```

---

## 四、模型保存与加载

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
ckpt = torch.load('model.pth')
model.load_state_dict(ckpt)
```

---

## 五、重要注意事项

### model.eval() vs torch.no_grad()

- `model.eval()`：改变某些层的行为（如 Dropout、BatchNorm）
- `torch.no_grad()`：防止计算被添加到梯度计算图中

**验证和测试时两者都要使用！**

---

## 六、常见错误及解决方案

### 6.1 设备不匹配
```python
# 错误：Tensor for * is on CPU, but expected them to be on GPU
# 解决：确保张量和模型在同一设备上
x = x.to('cuda')
```

### 6.2 维度不匹配
```python
# 错误：The size of tensor a (5) must match the size of tensor b (4)
# 解决：使用 transpose, squeeze, unsqueeze 调整维度
y = y.transpose(0, 1)
```

### 6.3 GPU 内存不足
```python
# 错误：CUDA out of memory
# 解决：减小 batch_size 或逐批处理数据
for d in data:
    out = model(d.to('cuda').unsqueeze(0))
```

### 6.4 张量类型不匹配
```python
# 错误：expected scalar type Long but found Float
# 解决：转换张量类型
labels = labels.long()
```

---

## 七、PyTorch 文档

官方文档：https://pytorch.org/docs/stable/

常用模块：
- `torch.nn` - 神经网络层
- `torch.optim` - 优化算法
- `torch.utils.data` - Dataset, Dataloader

---

## 八、扩展库

- **torchaudio** - 语音/音频处理
- **torchtext** - 自然语言处理
- **torchvision** - 计算机视觉
- **skorch** - scikit-learn + PyTorch

---

## 九、实用资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Huggingface Transformers](https://github.com/huggingface/transformers) - BERT, GPT 等
- [Fairseq](https://github.com/facebookresearch/fairseq) - 序列建模
- [PyTorch for NumPy Users](https://github.com/wkentaro/pytorch-for-numpy-users)
