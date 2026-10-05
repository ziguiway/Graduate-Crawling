# 对齐 Alignment

**一句话**：让两个原本对不上号的东西，对上号。

领域里两种用法，内核一样（建立对应关系），对的对象不同：

## 1. 表征对齐（feature / representation alignment）

- **对齐对象**：图 ↔ 词（两个向量空间）
- **例**：LLaVA Stage 1——训练后"猫图的特征向量 ≈ 'cat' 的词向量"，模型看到猫图像看到这个词。靠 59.5 万图文对 + 投影 W 学出来
- **判断**：宾语是空间 / 特征 / embedding / 模态

## 2. 价值对齐（value alignment / safety alignment）

- **对齐对象**：模型行为 ↔ 人类意图——人认为有害的，模型也认为有害、会拒绝
- **例**：InstructGPT / RLHF 一脉；ChatGPT 拒答有害指令。安全领域的 "alignment failure"、"alignment faking" 全是这个意思
- **判断**：宾语是人类意图 / 价值 / 偏好 / 安全

## 怎么区分

读论文时别背定义，直接问：**这里在对齐"什么"和"什么"？**

## 与安全方向的联系

LLaVA 的价值对齐只在**文本**上练过，图像通道只做了表征对齐、没做过价值对齐——把恶意指令藏进图片，可以绕过文本侧的 refusal 防线。多模态越狱的一大动机就在这个错位上。

## 在哪遇到过

- [[../论文阅读/LLM Safety/003-Visual Instruction Tuning (LLaVA)]] —— 首次弄懂；一篇论文里两种用法都出现
