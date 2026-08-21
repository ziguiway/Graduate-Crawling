---
title: Q&A：最小编辑距离在自动纠错中的作用
aliases:
  - 最小编辑距离在哪里起作用
tags:
  - NLP/拼写纠错
  - NLP/编辑距离
related:
  - "[[notes/NLP/自动拼写纠错/自动拼写纠错|自动拼写纠错]]"
---

# Q&A：最小编辑距离在自动纠错中的作用

## 问题描述

作业前面用 `edit_one_letter()`、`edit_two_letters()` 生成纠错候选，后面又实现了 `min_edit_distance()`。那么最小编辑距离到底在哪里参与了自动纠错？是不是用它直接选出距离最短的词？

## 回答

### 先看作业实际流程

```text
错误单词
  ↓
edit_one_letter / edit_two_letters
  ↓
候选词 ∩ vocab
  ↓
按 probs 的词频概率排序
  ↓
返回 top-n
```

`get_corrections()` 的实际逻辑是：

1. 如果输入词已经在词表中，直接保留；
2. 否则生成一次编辑候选；
3. 如果一次编辑候选中有词表内的词，就使用它们；
4. 如果没有，再生成两次编辑候选；
5. 对候选按语料概率排序。

所以，**本作业的自动纠错函数没有直接调用 `min_edit_distance()`**。

### 那它在哪里起作用？

它在 notebook 的第 4 部分单独起作用，用动态规划计算两个字符串之间的最小编辑代价：

```python
matrix, min_edits = min_edit_distance(source, target)
```

它还用于验证前面生成的候选：

```python
targets = edit_one_letter(source, allow_switches=False)
for target in targets:
    _, distance = min_edit_distance(source, target, 1, 1, 1)
```

如果 `target` 是一次编辑得到的词，那么验证结果应该是 1；两次编辑得到的词，结果应该是 1 或 2。这里的 `allow_switches=False` 是因为动态规划版本只实现了插入、删除、替换，没有实现相邻交换。

### 它和候选生成是什么关系？

前面的四种编辑函数是在**枚举**候选：

```text
删除一次、替换一次、插入一次、交换一次
→ 得到 E1(word)
```

因此 `edit_one_letter()` 已经隐含了“编辑距离为 1”这个概念；`edit_two_letters()` 已经隐含了“最多经过两次编辑”这个概念。

后面的 `min_edit_distance()` 则是用动态规划**精确计算**任意两个字符串的距离。

| 部分 | 作用 |
|---|---|
| `edit_one_letter()` | 枚举一次编辑能得到的候选 |
| `edit_two_letters()` | 枚举两次编辑能得到的候选 |
| `min_edit_distance()` | 计算两个字符串的精确最小编辑代价 |
| `get_corrections()` | 词表过滤后按词频概率排序 |

如果设计一个更完整的纠错系统，也可以先生成较大候选集，再用最小编辑距离过滤或参与打分。但这不是本次作业 `get_corrections()` 的实际实现。

## 一句话记忆

> 本作业中，编辑操作负责生成候选，`min_edit_distance()` 负责计算和验证字符串距离，`probs` 才负责最终排序；最小编辑距离没有直接在 `get_corrections()` 中调用。
