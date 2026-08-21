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

如果 `target` 是在关闭交换、每种操作成本都设为 1 时通过一次插入/删除/替换得到的不同字符串，那么验证结果应为 1；两次操作得到的结果可能重新测出 0、1 或 2。作业里出现 `eer -> eer` 的测试结果，就是两次操作互相抵消的例子。

### 重新核对：`edit_one`、`edit_two` 和最小编辑距离

这里不能把几个概念直接画等号。

- `edit_one_letter()`：枚举**执行一次允许的编辑操作**后的字符串。允许的操作包括删除、插入、替换，以及可选的相邻交换。
- `edit_two_letters()`：先做一次允许的编辑，再做一次允许的编辑。它表示“两步操作路径”，不保证最终的最小编辑距离恰好是 2，因为两次操作可能互相抵消。
- `min_edit_distance()`：对两个已经给定的字符串，用动态规划计算最小编辑代价。它不会生成候选词，也不会查词典。

还有一个重要区别：`min_edit_distance()` 只实现删除、插入、替换，没有实现相邻交换。因此打开 `allow_switches` 时，`edit_one_letter()` 生成的交换结果不能直接拿这个函数验证为距离 1。关闭交换后，普通的非原词结果才可以按这里的插入、删除、替换距离理解。

### 它在本作业里到底有没有用于自动纠错？

没有。`get_corrections()` 没有调用 `min_edit_distance()`。它的实际代码是：

```text
输入词在 vocab 中 → 直接建议输入词
否则：
    先生成 E1(word)，只保留 vocab 中的词
    E1 有结果 → 按概率排序并返回
    E1 没结果 → 再生成 E2(word)，只保留 vocab 中的词
```

所以你的疑问是成立的：在这条自动纠错流程里，`edit_one_letter()` / `edit_two_letters()` 已经负责按照“做几次编辑”搜索候选；再调用 `min_edit_distance()` 只能重新检查距离，不能生成新候选，也没有参与最终排序。

`min_edit_distance()` 在这个 notebook 里主要是两件事：

1. 单独练习字符串距离的动态规划算法；
2. 在后面的测试代码中验证候选生成函数，例如检查关闭交换后，`edit_one_letter()` 的结果距离为 1，`edit_two_letters()` 的结果可能为 0、1 或 2。

因此，当前先把它记成一个**独立的动态规划练习/验证工具**即可，不必强行说它已经参与了自动纠错主流程。只有当候选词来自整个词典或其他来源、而不是由 `edit_one` / `edit_two` 直接按编辑操作生成时，最小编辑距离才可能用来过滤或排序候选；这不是本 notebook 的实现。

## 待理解标记

- **状态：待理解，先搁置。**
- 当前疑问：既然 `edit_one_letter()` 已经通过一次编辑操作枚举候选，`edit_two_letters()` 已经通过两次编辑操作枚举候选，为什么还要计算 `min_edit_distance()`？它在这个作业的自动纠错流程中没有被调用，似乎是重复的。
- 已确认的代码事实：`min_edit_distance()` 不负责生成候选，也不参与 `get_corrections()` 的最终排序；它主要出现在独立的动态规划练习和候选验证代码中。
- 需要避免的误解：不能把 `edit_one_letter()` 无条件说成“标准编辑距离恰好为 1”，也不能把 `edit_two_letters()` 无条件说成“标准编辑距离恰好为 2”。前者可能包含相邻交换，后者的两次操作可能抵消；只有关闭交换并统一操作成本时，后面的验证才对应普通的插入/删除/替换距离。

## 一句话记忆

> 当前 notebook 中，`edit_one` / `edit_two` 是按编辑操作枚举候选，`min_edit_distance` 是对已给定的两个字符串计算最小代价；它们没有接入同一条自动纠错主流程，这个疑问先标为“待理解”。
