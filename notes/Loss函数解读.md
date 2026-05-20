# Loss 函数解读：如何判断模型训练得好不好

以 HW1 回归任务为例，科普回归任务最常用的几个评估指标。

---

## MSE（Mean Squared Error，均方误差）

回归任务最常用的损失函数。

**公式：**
$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_{pred} - y_{true})^2
$$

**怎么理解：**
- 每个样本：预测值和真实值差多少
- 平方一下（消除正负，放大大误差）
- 所有样本取平均

**特点：**
- 对大误差惩罚很大（平方效应）
- **单位不是原始单位**（因为平方了）
- 范围：[0, +∞)，越小越好

**代码：**
```python
nn.MSELoss(reduction='mean')
```

---

## RMSE（Root Mean Squared Error，均方根误差）

解决了 MSE 单位不一致的问题。

**公式：**
$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum(y_{pred} - y_{true})^2}
$$

**怎么理解：**
- 开根号后，**单位和原始数据一致**
- 直观含义：**"平均偏差多少"**

**举例（你的训练结果）：**
```
MSE = 1.20 → RMSE ≈ 1.10
→ 模型平均偏差约 1.1 个单位
```

**对比 target 分布：**
```
target 范围: 2.3 ~ 41.0  →  RMSE 1.1  ≈ 范围宽度的 2.8%
target 均值: 16.43        →  RMSE 1.1  ≈ 均值的 6.7%
```

---

## R²（R-squared / 决定系数）

**最重要的评价指标**，回答"模型比瞎猜好多少"。

**公式：**
$$
R^2 = 1 - \frac{\text{MSE}_{model}}{\text{MSE}_{baseline}}
$$

其中：
- **MSE_model**：你的模型的 MSE
- **MSE_baseline**：**"永远猜均值"** 的 MSE = 方差（variance）

**怎么理解：**
- R² = 0.98：模型解释了 98% 的方差
- R² = 0.00：模型还不如直接猜均值
- R² < 0.00：模型比瞎猜还差（可能过拟合或代码有 bug）

**R² 的含义：**

| R² | 含义 |
|----|------|
| 0.98 | 极好，解释了绝大部分方差 |
| 0.80 - 0.95 | 很好 |
| 0.50 - 0.80 | 还行，有改进空间 |
| 0.00 - 0.50 | 弱，baseline 都快追上你了 |
| < 0 | 出问题了 |

---

## MAE（Mean Absolute Error，平均绝对误差）

**公式：**
$$
\text{MAE} = \frac{1}{n}\sum|y_{pred} - y_{true}|
$$

**MSE vs MAE 对比：**

| | MSE/RMSE | MAE |
|---|---|---|
| 对大误差的态度 | **狠狠惩罚**（平方放大） | 一视同仁 |
| 单位 | RMSE 和原始单位一致 | 和原始单位一致 |
| 适用场景 | 想避免大误差时 | 所有误差同样重要时 |
| 数学性质 | 处处可导（梯度好用） | 0处不可导 |

---

## 怎么才算"好"？

回归任务**没有绝对标准**，必须结合目标值的范围判断。

**COVID-19 数据分布：**

| 指标 | 值 |
|------|-----|
| target 最小值 | 2.34 |
| target 最大值 | 40.96 |
| target 均值 | 16.43 |
| target 方差 | 58.03（baseline MSE） |
| target 标准差 | 7.62（baseline RMSE） |

**你的模型结果这么看：**

| 指标 | 你的值 | 评价 |
|------|--------|------|
| MSE | 1.20 | — |
| RMSE | **1.10** | 平均偏差 1.1，target 范围 2.3~41 |
| R² | **0.979** | 解释了 97.9% 的方差 |

**和 baseline 对比（最直观）：**
```
"永远猜均值" 的误差:  RMSE = 7.62
你的模型误差:          RMSE = 1.10

→ 你的模型误差是瞎猜的 1/7，好太多了 ✅
```

**判断 checklist：**

1. **看 R²** — 越接近 1 越好，0.9+ 算不错
2. **看 RMSE** — 和 target 范围对比，看是否可接受
3. **看 train vs dev loss** — 两条线是否接近（防过拟合）
4. **对比 baseline** — 比"永远猜均值"好多少

---

## 早停（Early Stopping）

不要把早停当成"没训练够"。**早停是保护机制**——当模型在验证集上不再进步时及时停下来，**防止过拟合**。

```
early_stop 设置技巧：
- 小数据集：early_stop = 20~50
- 大数据集：early_stop = 5~10
- 设为 2 → 太敏感了，容易过早停
- 设为 100+ → 除非你很闲
```

---

## 学习曲线怎么看

训练时记录 train loss 和 dev loss，画在一张图上是**最重要的诊断工具**。

**正常拟合：**
```
train loss ↘    dev loss ↘    两条线接近 → 正常 ✅
```

**过拟合（dev loss 反弹）：**
```
train loss ↘（继续降）   dev loss ↘→↗（开始反弹）
→ 差距大 → 过拟合 ⚠️
```

**欠拟合（降不下去）：**
```
train loss →（不动）   dev loss →（不动）
→ 都降不下去 → 欠拟合 ⚠️
```

---

## 一键评估函数

```python
def evaluate_model(model, dv_set, device):
    model.eval()
    preds, targets = [], []
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    mse = ((preds - targets) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(preds - targets).mean()
    r2 = 1 - mse / targets.var()

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
```
