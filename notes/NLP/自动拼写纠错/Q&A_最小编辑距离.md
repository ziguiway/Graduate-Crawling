---
title: Q&A：最小编辑距离在自动纠错中的作用
aliases:
  - 最小编辑距离在哪里起作用
tags:
  - NLP/拼写纠错
  - NLP/编辑距离
status: 待理解
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

### 最直白的理解：它到底怎么用于自动纠错？

把 `min_edit_distance()` 想成一把尺子：它用来量“用户输入的错词”和“词典里的候选词”之间差几步。

例如用户输入：

```text
dys
```

系统可以检查词典里的候选词：

```python
# 把插入、删除、替换都按 1 步计算
min_edit_distance("dys", "days", 1, 1, 1) = 1   # 插入 a
min_edit_distance("dys", "dye", 1, 1, 1)  = 1   # 把 s 换成 e
```

于是 `days` 和 `dye` 都可能是纠正结果。两者距离一样时，再比较词频：哪个词在语料里更常见，就优先推荐哪个。于是可能得到：

```text
输入 dys
→ 候选 days、dye
→ 两个都只差 1 步
→ days 词频更高
→ 推荐 days
```

如果真的把 `min_edit_distance()` 接到纠错里，流程就是：

```text
输入错词
  ↓
对词典中的每个词计算最小编辑距离
  ↓
保留距离为 1 或 2 的词
  ↓
距离相同时，按词频排序
  ↓
返回最可能的词
```

但这个 notebook 没有这样逐个计算。它采用了一个更直接的办法：`edit_one_letter()` 直接把“一步能到的所有字符串”枚举出来，所以这些字符串天然就是编辑距离 1 的候选；`edit_two_letters()` 同理。也就是说，前面的代码相当于“直接找距离为 1 或 2 的词”，后面的 `min_edit_distance()` 才是“拿尺子测两个词的距离”。

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

## 待理解标记

- **状态：待理解，先搁置。**
- 当前疑问：既然 `edit_one_letter()` 已经生成编辑距离为 1 的候选，`edit_two_letters()` 已经生成编辑距离为 2 的候选，为什么还要计算 `min_edit_distance()`？它在这个作业的自动纠错流程中似乎是重复的。
- 暂时保留的观察：`min_edit_distance()` 目前主要出现在独立的动态规划练习和候选验证代码中；它是否有超出“重新验证距离”的实际作用，等以后结合新的场景再理解。

## 一句话记忆

> 本作业中，`edit_one_letter()` / `edit_two_letters()` 已经完成了候选生成，因此 `min_edit_distance()` 对自动纠错不是必需的；它主要是独立的动态规划练习和候选验证工具。
