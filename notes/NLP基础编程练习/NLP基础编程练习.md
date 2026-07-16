---
title: NLP 基础编程练习
date: 2026-07-16
tags:
  - NLP
  - 编程练习
  - 导师布置
status: 进行中
source: https://mp.weixin.qq.com/s/H2BTy2cKBzjHjKLXzJKbWQ
teacher_page: http://hlt.suda.edu.cn/index.php/New-stu-training
---

# NLP 基础编程练习

> [!info] 来源
> - 公众号文章：李正华《我设计的 NLP 基础编程练习》（2025-07-28 完稿）—— [公众号链接](https://mp.weixin.qq.com/s/H2BTy2cKBzjHjKLXzJKbWQ)
> - 老师主页（题目、数据、视频、课件都在这）：[New-stu-training](http://hlt.suda.edu.cn/index.php/New-stu-training)
> - 题目（课程）主页：[cip-2015-fall](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/)
> - 导师布置，要求有空就开始做。

## 这套练习是干什么的

李老师设计的这套 NLP 基础编程练习有两个用途：

1. **硕士招生考察**（1–2 周，每天 1–4 小时）：通过持续做编程任务 + 发日志，看学生的沟通表达、逻辑思维、编程基础。
2. **硕士生打基础**：被录取后，在大四下 / 暑假 / 入学后一两个月做完，作为科研起步的基础。

> [!quote] 李老师对"慢"的态度
> "慢即是快，少即是多。慢慢地一点一点搞明白的过程，虽然感觉很艰难，进度会比较慢，但是理解会很深刻，会有很多自己的体会，将来会激发自己的科研灵感。"

> [!warning] 两条禁忌
> - 一遇到困难就去看别人的代码 —— 不好。
> - 在网上找多种不同资料（包括视频）—— 也不好。材料多了脑子会乱，给的讲义和视频已包含所有必要信息，要沉下心琢磨、看数据、推公式。

## 任务总览

共 **11 个核心作业** + **4 个后续扩展**。任务 1–7 是传统机器学习方法（CRF 之前），任务 8 起进入神经网络 / 深度学习。任务主线是**词性标注（POS Tagging）**，从简单到复杂逐步演进模型。

| #   | 任务              | 模型类型                     | 局部/全局 | 关键技术                                                  | 状态    |
| --- | --------------- | ------------------------ | ----- | ----------------------------------------------------- | ----- |
| 1   | 分字（GB / UTF8）   | —                        | —     | 汉字编码、BOM、大小字节序                                        | ⬜ 未开始 |
| 2   | 最大匹配分词          | 贪心                       | —     | 词典分词、P/R/F 评价                                         | ⬜ 未开始 |
| 3   | 有监督 HMM 词性标注    | 生成式                      | —     | 贝叶斯、概率平滑、Viterbi 解码（DP）                               | ⬜ 未开始 |
| 4   | 线性模型词性标注        | 判别式（Averaged Perceptron） | 局部    | 稀疏特征向量、特征模板、延迟更新 v、partial feature                    | ⬜ 未开始 |
| 5   | 最大熵词性标注         | 对数线性（局部）                 | 局部    | 梯度推导、模拟退火步长调整                                         | ⬜ 未开始 |
| 6   | 全局线性模型（GLM）词性标注 | 判别式                      | 全局    | POS tag bigram、动态规划解码（同 Viterbi）                      | ⬜ 未开始 |
| 7   | CRF 词性标注        | 对数线性（全局）                 | 全局    | Viterbi、forward-backward、logsumexp、梯度推导               | ⬜ 未开始 |
| 8   | FFN 词性标注        | 神经网络                     | 局部    | Word embedding、MLP、softmax、**手写 back-prop**（建议 numpy） | ⬜ 未开始 |
| 9   | FFN-CRF 词性标注    | 神经网络 + CRF               | 全局    | 在 FFN 基础上加 transition matrix，numpy 手写 BP              | ⬜ 未开始 |
| 10  | BiLSTM 词性标注     | 神经网络                     | 局部    | BiLSTM 全局上下文表示、学习 PyTorch                             | ⬜ 未开始 |
| 11  | BiLSTM-CRF 词性标注 | 神经网络 + CRF               | 全局    | PyTorch CRF 层（建议自己实现 forward 算 loss）                  | ⬜ 未开始 |

> [!tip] 进度标记约定
> 每个任务的状态用：⬜ 未开始 / 🔄 进行中 / ✅ 已完成。开始做一个任务时，在对应任务详情小节末尾追加「日志」记录进展和遇到的困难（呼应李老师"发日志讲进展和困难"的要求）。

## 数据与共用资料

### 参考书目（老师推荐）

- [Speech and Language Processing（Jurafsky, SLP3）](https://web.stanford.edu/%7Ejurafsky/slp3/) —— [中文翻译](https://www.kancloud.cn/drxgz/slp20201230#/dashboard)（==强烈推荐==）
- [Neural Networks and Deep Learning（Nielsen）](http://neuralnetworksanddeeplearning.com/) —— ==李正华强烈推荐，看完前 3 章差不多==
- [吴恩达深度学习（带中文字幕）](https://mooc.study.163.com/university/deeplearning_ai#/c)
- Chris Manning. 2005. *统计自然语言处理基础*
- 宗成庆. 2008. *统计自然语言处理*
- 李航. 2012. *统计学习方法*
- 神经网络与深度学习（教材）

### CoNLL 格式说明

老师主页明确：每个词占一行，每行的第 **2** 列为当前词语，第 **4** 列为当前词的词性，第 **7** 列为当前词的中心词的序号，第 **8** 列为当前词语与中心词的依存关系。句子与句子之间以空行间隔。

> [!example] 词性标注示例
> 输入：`严守一 把 手机 关 了`
> 输出：`严守一/NR 把/P 手机/NN 关/VV 了/SP`

### 词性标注数据（共用）

所有词性标注作业（作业 3–11）都用同一份数据，UTF8 编码：

| 数据集 | 用途 | 规模 | 下载 |
|---|---|---|---|
| 小数据集 | 入门练手 | 训练 803 句 / 开发 1910 句 | [data.tar.gz](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/6-ngram-language-model/data.tar.gz) |
| 大数据集 | 正式实验 | 训练 16091 句 / 开发 803 句 / 测试 1910 句 | [ctb5-postagged.tar.gz](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/ctb5-postagged.tar.gz) |
| 分词数据 | 作业 2 专用 | — | [data.conll](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/5-chinese-word-segmentation/data.conll) |

> [!tip] 从小数据集开始
> 入门阶段先用小数据集（803 句训练），跑通整个 pipeline；再换大数据集看准确率变化。

## 逐个任务详情

### 作业 1：分字（汉字编码 GB / UTF8）

- **目标**：搞清楚汉字在计算机中如何保存、如何把句子里的每个字切出来。
- **题目要求**：给定文件，将文件中的句子按字（字符）切分，字符间用空格隔开。用 ==C/C++== 实现。Python（3.0）可直接用 split 处理 UTF8 字符串，也试一下，对比结果。
- **UTF-8 编码规则**（页面给出，必记）：
  - 1 字节：`0xxxxxxx`
  - 2 字节：`110xxxxx 10xxxxxx`
  - 3 字节：`1110xxxx 10xxxxxx 10xxxxxx`
  - 4 字节：`11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`
  - RFC 3629（2003）已废除 5–6 字节，==只用考虑 1–4 字节==。
- **易踩坑**：UTF8 文件开头可能有 BOM（`0xFE 0xFF` 大字节序 Big-Endian，`0xFF 0xFE` 小字节序 Little-Endian），通常直接删除即可；分字后出现乱码再去了解大小字节序。
- **李老师的故事**：他大三（2004 秋）刚接触 NLP 时这个问题困扰了他几个月，后来靠看师兄姐代码搞清楚 GB，再看别的代码搞清楚 UTF8。

**资料**
- 视频（2022 春 IR 课程）：
  - [作业1 低画质](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-low-quality.mp4)
  - [作业1-part1 高画质](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-part-1.mp4) ｜ [作业1-part2 高画质](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-part-2.mp4)
- 图片：[图1](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-figure-1.jpg) ｜ [图2](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-figure-2.jpg)
- 数据：[文件:Sentence.txt（UTF8）](http://hlt.suda.edu.cn/index.php/%E6%96%87%E4%BB%B6:Sentence.txt)
- 参考资料：[文件:Chinese-encoding.pdf](http://hlt.suda.edu.cn/index.php/%E6%96%87%E4%BB%B6:Chinese-encoding.pdf)
- 不同编码示例文件：[example.tar.gz](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/2-chinese-encoding/example.tar.gz)（可用 `hexdump` 查看，也可自己生成不同编码的文件）

### 作业 2：最大匹配分词

- **算法**：简单的贪心。给定词典，从左向右扫描句子切成词序列。
- **易踩坑**：==评价部分== —— 精确率 P、召回率 R、F 值的计算。
- **正确实验结果**（用来验证自己程序对不对）：
  - 正确识别的词数：**20263**
  - 识别出的总体个数：**20397**
  - 测试集中的总体个数：**20454**
  - 正确率：**0.99343**
  - 召回率：**0.99066**
  - F 值：**0.99204**
- **李老师的故事**：大四上《中文信息处理》课（刘秉权老师），在机房做了整整一下午，当时用 C++，特别有成就感。

**资料**
- 视频（2022 春 IR 课程）：
  - [作业3 低画质](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/word-seg-max-match-low-quality.mp4) ｜ [作业3 高画质](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/word-seg-max-match.mp4)
- 图片：[图](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/word-seg-max-match.jpg)
- 数据：
  - 词典：[文件:Dict.txt](http://hlt.suda.edu.cn/index.php/%E6%96%87%E4%BB%B6:Dict.txt)
  - 待分词：[文件:Sentence.txt](http://hlt.suda.edu.cn/index.php/%E6%96%87%E4%BB%B6:Sentence.txt)
  - 正确答案（人工标注，对比算 P/R/F）：[文件:Answer.txt](http://hlt.suda.edu.cn/index.php/%E6%96%87%E4%BB%B6:Answer.txt)
  - 正向最大匹配的预测结果（程序写对应该和这个一模一样）：[文件:Out.txt](http://hlt.suda.edu.cn/index.php/%E6%96%87%E4%BB%B6:Out.txt)
- 参考课件：[最大匹配（ppt）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/5-chinese-word-segmentation/max-match.ppt)

### 作业 3：有监督 HMM 词性标注

- **模型**：隐马尔可夫模型（HMM），经典、简单、有效。深度学习前主流 ASR 都基于 HMM。
- **只做有监督版**（含少量公式推导，贝叶斯公式的典型应用）。
- **关键点**：
  - 概率平滑（好玩且有用）
  - ==Viterbi 解码==：核心，典型动态规划算法，找概率最高的词性序列
- **延伸**：无监督 HMM 是无监督学习的经典例子（EM 算法），等以后有需要再学，学习曲线很好。
- **为什么要学经典知识**（李老师的理由）：
  1. 有些论文基于经典知识，不学读不懂。
  2. 学完可模仿借鉴其做法设计自己的方法。
  3. 了解创造过程能让研究思路更宽、激发灵感。
  4. 学习老经典 = 学历史，关乎底蕴和视野；深度学习/大模型之后是什么谁也不知道，盛极必衰，知识技能仅限近一二十年新东西的人，创造力和品味不会出类拔萃。

**资料**
- 视频（2022 春 IR 课程，分低画质 / 高画质 / 图片三套）：
  - 低画质：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1.mp4) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2.mp4) ｜ [第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3.mp4) ｜ [第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4.mp4)
  - 高画质：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1-hd.mp4) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2-hd.mp4) ｜ [第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3-hd.mp4) ｜ [第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4-hd.mp4)
  - 图片（截图）：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1.jpg) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2.jpg) ｜ [第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3.jpg) ｜ [第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4.jpg)
- 参考课件：
  - [Collins 教授课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging/collins-tagging.pdf)
  - [李正华老师课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging/main.pdf)
  - [理解 HMM 的 Viterbi（pptx）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging/HMM-v2.pptx)
  - [HMM 模型中极大似然估计的由来（公式推导，pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging-MLE/main.pdf)
- 数据：见上方 [[#词性标注数据（共用）]]

### 作业 4：基于线性模型的词性标注

- **模型**：Averaged Perceptron（平均感知器）。Perceptron 本质是没有非线性变换的一层神经网络 —— 所以线性模型和深度学习有联系。
- **要点**（页面标注）：判别模型、partial feature。
- **核心概念**：
  - 稀疏特征向量 `f(x)`：人觉得对决策有帮助的信息，NLP 里是拼凑的字符串
  - 特征空间、特征模板
  - `f(x)` 与权重向量 `w` 点积得到打分
- **两个重要优化技术**（李老师花很多精力看别人代码学会的）：
  - ==训练阶段延迟更新 v==（累积的特征权重向量）
  - ==partial feature==
  - 这两个对训练和预测速度影响很大。
- **价值**：搞明白线性模型，才会真正理解为什么"上下文表示能力 representation"是深度学习的最大优势 —— 传统稀疏特征向量很难有效表示一个句子，更别说文档。

**资料**
- 参考课件：[李正华老师课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/9-linear-model/main2.pdf)
- 视频（2022 春 IR 课程）：
  - [第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-1.mp4) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-2.mp4) ｜ [第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-3.mp4) ｜ [第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-4.mp4) ｜ [第5部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-5.mp4)
- 图片：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-1.jpg) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-2.jpg) ｜ [第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-3.jpg) ｜ [第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-4.jpg) ｜ [第5部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-5.jpg)
- 数据：见 [[#词性标注数据（共用）]]

### 作业 5：基于最大熵模型的词性标注

- **模型**：最大熵 = 对数线性模型，在打分基础上给出类别概率。
- **要点**（页面标注）：梯度下降方法，Adam 优化。
- **与作业 4 的区别**：训练目标函数（概率形式）不同。
- **关键点**：
  - 根据训练目标函数推导针对权重向量 `w` 的梯度 —— 典型求导过程，必须掌握。
  - ==模拟退火== 思想的步长调整（深度学习的 Adam 本质也是步长调整策略，可直接用）。
- **李老师的故事**：2008 年通过读 Ryan McDonald 的 MSTParser（Java）真正搞懂线性模型；2010 年底张梅山师兄在白板上推导了一遍最大熵公式（目标函数、求导、模拟退火），李老师才意识到"原来最大熵很简单，可以自己写出来"——第一性原理的体现。
- **李老师的存疑**：如何由"最大化熵值"得到对数线性模型形式（涉及对偶优化），看了李航《机器学习》仍没搞明白，打算继续研究。

**资料**
- 参考课件：
  - [李正华老师课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/10-maxent-loglinear/main.pdf)
  - [Collins 教授课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/10-maxent-loglinear/collins-loglinear.pdf)
- 视频（2022 春 IR 课程）：
  - [第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/Maximum-entropy-1.mp4) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/Maximum-entropy-2.mp4)
- 图片：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/Maximum-entropy-1.jpg) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/Maximum-entropy-2.jpg)
- 数据：见 [[#词性标注数据（共用）]]

### 作业 6：基于全局线性模型（GLM）的词性标注

- **对比作业 4**：作业 4 是局部模型（一次判一个词的词性），GLM 是全局模型，额外考虑相邻词性联系（POS tag bigrams），刻画完整词性序列的分值。
- **核心**：一个动态规划解码算法，和 HMM 的 Viterbi 基本一致。

**资料**
- 参考课件：[李正华老师课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/11-global-linear-model/main.pdf)
- 数据：见 [[#词性标注数据（共用）]]

### 作业 7：基于 CRF 的词性标注

- **模型**：CRF = 对数线性 + 全局，刻画完整词性序列的概率。
- **要点**（页面标注）：全局概率、期望、Forward-backward 结合、viterbi 解码。
- **可理解为**：==GLM + 最大熵的结合体==。搞明白前两者，CRF 就很容易。
- **核心**：
  - 两个动态规划算法：Viterbi 解码、forward-backward 求期望
  - 针对权重向量 `w` 的梯度推导 —— 很值得搞明白
  - 实现 forward-backward 时，sumexp 溢出问题用 ==logsumexp== 解决

**资料**
- 参考课件：[李正华老师课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/12-crf/main.pdf)
- 数据：见 [[#词性标注数据（共用）]]

### 作业 8：基于前馈神经网络（FFN）的词性标注

- **从这里开始进入神经网络 / 深度学习**。
- **要点**（页面标注）：==必须自己实现前向计算 loss，和 backpropagation==。
- **推荐入门材料**：Nielsen《Neural Networks and Deep Learning》（2015），很薄，看完前 3 章就够了解基础，后 3 章是高阶技巧。优点：薄、直入主题。
- **模型**：和作业 5 最大熵基本一致，都是局部模型。给定一个词 + 周围 4 个词（窗口 5），拼接 word embedding → MLP（2–3 层，带 ReLU 等非线性）→ softmax 预测概率。注意：最大熵里"分值得概率"的过程就是 softmax。Word embedding 可用预训练的 GloVe。
- **最重要两点**：
  1. 学习深度学习基本知识，入门
  2. ==把 back-propagation 算法完全搞明白并自己实现==（BP 本质是动态规划 / 求导链式法则）
- **强烈建议**：==用 numpy 实现，不要用 PyTorch==（为了真正搞懂 BP）。

**资料**
- 入门书：[Neural Networks and Deep Learning（Nielsen）](http://neuralnetworksanddeeplearning.com/)（看完前 3 章即可完成本任务）
- 视频：[吴恩达深度学习（带中文字幕）](https://mooc.study.163.com/university/deeplearning_ai#/c)
- 李老师 NLP-DL 公开课第十三章：
  - [第1节：从离散特征到连续稠密向量表示](http://hlt.suda.edu.cn/~zhli/NLP-DL/13.1.mp4)
  - [第2节：表示学习](http://hlt.suda.edu.cn/~zhli/NLP-DL/13.2.mp4)
  - [第3节：序列标注问题](http://hlt.suda.edu.cn/~zhli/NLP-DL/13.3.mp4)
  - [第4节：句法树解析问题](http://hlt.suda.edu.cn/~zhli/NLP-DL/13.4.mp4)
- 数据：见 [[#词性标注数据（共用）]]

### 作业 9：基于 FFN-CRF 的词性标注

- **要点**（页面标注）：仍然自己实现前向计算 loss，和 backpropagation。
- **提示**（页面给出）：==将神经网络输出看成发射矩阵，之后加上转移矩阵==。
- 在 FFN 局部预测分值基础上，加入 POS tag bigram 的分值（transition matrix）。
- ==完全可以在 numpy 基础上做==，用自己实现的 back-propagation 算法。

**资料**
- 入门书、视频同作业 8（Nielsen 书 + 吴恩达视频 + 李老师第 13 章）
- 数据：见 [[#词性标注数据（共用）]]

### 作业 10：基于 BiLSTM 的词性标注

- **要点**（页面标注）：可以利用 PyTorch 自带的。==Dropout 等的使用，是关键==。
- **仍然属于局部模型**，但先通过 BiLSTM 对整个句子做全局表示，每个词的表示都包含全局上下文信息 —— 不需要像前面的模型拼接窗口内 embedding。
- **BiLSTM 实现麻烦**，但 PyTorch 有现成模块。
- **目的**：1）学 BiLSTM 原理；2）学 PyTorch。

**资料**
- 同作业 8 的入门材料
- 数据：见 [[#词性标注数据（共用）]]

### 作业 11：基于 BiLSTM-CRF 的词性标注

- PyTorch 中也有现成的 CRF 层可用。
- **建议**：==自己实现一下==，本质就是写一个 forward 算法算出 loss，然后自动求导机制完成其他工作。
- **参考代码**（老师主页给出）：github 已有代码，不同同学的代码可以看不同的 branch —— [SUDA-LA/CIP](https://github.com/SUDA-LA/CIP)（==仅作对照参考，不要直接抄==）

**资料**
- GitHub 仓库：[SUDA-LA/CIP](https://github.com/SUDA-LA/CIP)
- 数据：见 [[#词性标注数据（共用）]]

## 后续扩展（自主学习，时间允许时）

老师主页列出的 6 个扩展方向：

1. **基于图的依存句法分析**：直接用神经网络实现即可，Biaffine Parser 框架。重点：==Eisner 动态规划解码算法==（看李老师的 COLING-2014 tutorial）；进而可扩展到 TreeCRF，将 Eisner 算法扩展为 inside 算法。对应 ACL-2020 论文：Yu Zhang et al.
2. **基于转移的依存句法分析**：了解一下转移系统。
3. **Seq2Seq (RNN) NMT with attention**：了解一下语言生成。
4. **Transformer NMT**：技术细节很多（呼应我已学的 [[Transformer]]）。
5. **无监督学习**：HMM-EM、VAE。
6. **ELMo/BERT 的原理**。

**相关论文与代码**
- Yu Zhang, Houquan Zhou, Zhenghua Li. *IJCAI-2020. Fast and Accurate Neural CRF Constituency Parsing.*
- Yu Zhang, Zhenghua Li, Min Zhang. *ACL-2020. Efficient Second-Order TreeCRF for Neural Dependency Parsing.*
- 代码仓库：[supar](https://github.com/yzhangcs/parser)（张宇构建并持续维护）
- 老师主页的参考代码仓库：[SUDA-LA/CIP](https://github.com/SUDA-LA/CIP)（不同 branch 是不同同学的实现）

## 学习方法提醒（李老师原话提炼）

> [!quote] 学东西最好的方式
> 直接、简单、接近本质地告诉我"是什么、怎么工作"。"为什么这么做"可以自己慢慢琢磨。很多东西其实只要知道"是什么、大概怎么工作"就够了；等下一次接触到，再根据需求逐渐深入。

> [!quote] 不怕犯错
> 不怕犯错，从错误中学习，这是学习新东西最有效率的途径。凡事要大胆去做、去尝试。

> [!quote] 心态
> 世界上很多东西看似很难，其实如果能直入主题地去学习，一点一点学，耐心，不要对自己要求太高（降低期望），都不难。不能贪心，不能一下子学很多东西。在一段充裕的时间内，学习少量的东西。让学习过程更有乐趣，而不是只看结果，不过度关注进度。==少即是多，慢即是快。==

## 与已学内容的关联

- 作业 8 起进入深度学习，与我正在学的 [[机器学习基础]]、[[自注意力机制]]、[[Transformer]] 是一条线。
- 后续扩展 1（Transformer-encoder）正好接上我刚学完的 Transformer 第 7 章 —— 可以把词性标注当作 Transformer 的第一个实战任务。
- 作业 8 的 back-propagation 与我后续要学的训练技巧（梯度下降、反向传播）直接相关。

## 我的理解

这套练习最打动我的是它的**渐进式设计**：从最底层的字节编码（汉字怎么存）开始，一路走到 BiLSTM-CRF，再到 Transformer / BERT。每一站都是在前一站的基础上加一点点新东西（局部→全局、线性→对数线性、手工特征→embedding、前馈→循环→注意力），没有一步是"跳"上去的。这正好治我这种"看大模型看多了觉得老东西没用"的浮躁。

李老师反复强调的两点我要记牢：
1. **沉下心用给的材料，别到处找资料** —— 材料多了脑子会乱。我之前学 Transformer 时确实有这个毛病，B 站、知乎、原论文一起看，反而越看越糊涂。
2. **手写 > 调包** —— 作业 8 明确要求 numpy 手写 BP，作业 11 建议自己实现 CRF forward。只有手写过一遍，才会真正理解 PyTorch 那一行 `loss.backward()` 背后发生了什么。

接下来我先从**作业 1 分字**开始，用 C 语言做（顺便补一下 C 的基础，研究生阶段肯定用得上）。每天发日志记录进展和困难，按李老师的要求来。
