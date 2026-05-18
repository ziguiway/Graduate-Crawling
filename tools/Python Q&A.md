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
