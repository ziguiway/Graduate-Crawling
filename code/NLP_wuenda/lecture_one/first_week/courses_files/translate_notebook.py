import json
import copy

input_path = "/Users/zhengshuang/Documents/code/Graduate-Crawling/code/NLP_wuenda/lecture_one/first_week/courses_files/C1_W1_Assignment.ipynb"
output_path = "/Users/zhengshuang/Documents/code/Graduate-Crawling/code/NLP_wuenda/lecture_one/first_week/courses_files/C1_W1_Assignment_中文.ipynb"

with open(input_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 翻译映射：原文前几个字符 -> 翻译后的完整 source 列表
translations = {}

# Cell 0: 标题和欢迎
translations["# Assignment 1: Logistic Regression"] = [
    "# 作业 1：逻辑回归\n",
    "欢迎来到本专项课程的第一周。你将学习逻辑回归。具体来说，你将实现用于推文情感分析的逻辑回归。给定一条推文，你需要判断它是积极情感还是消极情感。你将完成以下内容：\n",
    "\n",
    "* 学习如何从文本中提取逻辑回归所需的特征\n",
    "* 从零开始实现逻辑回归\n",
    "* 将逻辑回归应用于自然语言处理任务\n",
    "* 测试你的逻辑回归模型\n",
    "* 进行错误分析\n",
    "\n",
    "我们将使用一个推文数据集。希望你能达到 99% 以上的准确率。\n",
    "运行下面的代码单元来加载所需的包。"
]

# Cell 1: Import functions and data
translations["## Import functions and data"] = [
    "## 导入函数和数据"
]

# Cell 3: Imported functions
translations["### Imported functions"] = [
    "### 已导入的函数\n",
    "\n",
    "下载本作业所需的数据。请查看 [twitter_samples 数据集文档](http://www.nltk.org/howto/twitter.html)。\n",
    "\n",
    "* twitter_samples：如果你在本地电脑上运行此 notebook，需要使用以下命令下载：\n",
    "```Python\n",
    "nltk.download('twitter_samples')\n",
    "```\n",
    "\n",
    "* stopwords：如果你在本地电脑上运行此 notebook，需要使用以下命令下载：\n",
    "```python\n",
    "nltk.download('stopwords')\n",
    "```\n",
    "\n",
    "#### 导入 utils.py 文件中提供的一些辅助函数：\n",
    "* `process_tweet()`：清洗文本，将其分词为独立单词，移除停用词，并将单词转换为词干。\n",
    "* `build_freqs()`：统计'语料库'（全部推文集合）中每个单词与积极标签 '1' 或消极标签 '0' 关联的次数，然后构建 `freqs` 字典，其中每个键是一个 (单词, 标签) 元组，值是该单词在推文语料库中出现的频率计数。"
]

# Cell 7: Prepare the data
translations["### Prepare the data"] = [
    "### 准备数据\n",
    "* `twitter_samples` 包含 5000 条积极推文的子集、5000 条消极推文的子集，以及包含 10000 条推文的完整集合。\n",
    "    * 如果你使用全部三个数据集，将会引入积极推文和消极推文的重复。\n",
    "    * 你只需选择 5000 条积极推文和 5000 条消极推文。"
]

# Cell 9: Train test split
translations["* Train test split: 20% will be in the test set"] = [
    "* 训练集/测试集划分：20% 作为测试集，80% 作为训练集。\n"
]

# Cell 11: Create numpy array of labels
translations["* Create the numpy array of positive labels and negative labels."] = [
    "* 创建积极标签和消极标签的 numpy 数组。"
]

# Cell 13: Create frequency dictionary
translations["* Create the frequency dictionary using the imported `build_freqs()` function."] = [
    "* 使用导入的 `build_freqs()` 函数创建频率字典。\n",
    "    * 我们强烈建议你打开 `utils.py` 并阅读 `build_freqs()` 函数，以理解它的作用。\n",
    "    * 要查看文件目录，请转到菜单并点击 File->Open。\n",
    "\n",
    "```Python\n",
    "    for y,tweet in zip(ys, tweets):\n",
    "        for word in process_tweet(tweet):\n",
    "            pair = (word, y)\n",
    "            if pair in freqs:\n",
    "                freqs[pair] += 1\n",
    "            else:\n",
    "                freqs[pair] = 1\n",
    "```\n",
    "* 注意外层 for 循环遍历每条推文，内层 for 循环遍历推文中的每个单词。\n",
    "* `freqs` 字典就是正在构建的频率字典。\n",
    "* 键是元组 (单词, 标签)，例如 (\"happy\",1) 或 (\"happy\",0)。每个键对应的值是单词 \"happy\" 与积极标签关联的次数，或与消极标签关联的次数。"
]

# Cell 15: Expected output
translations["#### Expected output"] = [
    "#### 预期输出\n",
    "```\n",
    "type(freqs) = <class 'dict'>\n",
    "len(freqs) = 11346\n",
    "```"
]

# Cell 17: Process tweet
translations["### Process tweet"] = [
    "### 处理推文\n",
    "给定的 `process_tweet()` 函数将推文分词为独立单词，移除停用词，并应用词干提取。"
]

# Cell 19: Expected output for process tweet
translations["#### Expected output\n```\nThis is an example of a positive tweet:"] = [
    "#### 预期输出\n",
    "```\n",
    "This is an example of a positive tweet: \n",
    " #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)\n",
    " \n",
    "This is an example of the processes version: \n",
    " ['followfriday', 'top', 'engag', 'member', 'commun', 'week', ':)']\n",
    "```"
]

# Cell 21: Part 1 Logistic regression / Sigmoid
translations["# Part 1: Logistic regression"] = [
    "# 第 1 部分：逻辑回归\n",
    "\n",
    "\n",
    "### 第 1.1 节：Sigmoid 函数\n",
    "你将学习使用逻辑回归进行文本分类。\n",
    "* Sigmoid 函数定义如下：\n",
    "\n",
    "$$ h(z) = \\frac{1}{1+\\exp^{-z}} \\tag{1}$$\n",
    "\n",
    "它将输入 'z' 映射到 0 到 1 之间的值，因此可以将其视为概率。\n",
    "\n",
    "<div style=\"width:image width px; font-size:100%; text-align:center;\"><img src='../tmp2/sigmoid_plot.jpg' alt=\"alternate text\" width=\"width\" height=\"height\" style=\"width:300px;height:200px;\" /> 图 1 </div>"
]

# Cell 23: Instructions sigmoid
translations["#### Instructions: Implement the sigmoid function"] = [
    "#### 说明：实现 sigmoid 函数\n",
    "* 你需要让这个函数在 z 是标量或数组时都能正常工作。"
]

# Cell 25: Hints (sigmoid) - 保持 HTML 结构，翻译文字
translations["<details>    \n<summary>\n    <font size=\"3\" color=\"darkgreen\"><b>Hints</b></font>"] = [
    "<details>    \n",
    "<summary>\n",
    "    <font size=\"3\" color=\"darkgreen\"><b>提示</b></font>\n",
    "</summary>\n",
    "<p>\n",
    "<ul>\n",
    "    <li><a href=\"https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html\" > numpy.exp </a> </li>\n",
    "\n",
    "</ul>\n",
    "</p>\n",
    "\n"
]

# Cell 29: Logistic regression regression and sigmoid
translations["### Logistic regression: regression and a sigmoid"] = [
    "### 逻辑回归：回归 + sigmoid\n",
    "\n",
    "逻辑回归在普通线性回归的基础上，对线性回归的输出应用 sigmoid 函数。\n",
    "\n",
    "线性回归：\n",
    "$$z = \\theta_0 x_0 + \\theta_1 x_1 + \\theta_2 x_2 + ... \\theta_N x_N$$\n",
    "注意 $\\theta$ 值是\"权重\"。如果你学习过深度学习专项课程，我们用 `w` 向量来表示权重。在本课程中，我们使用不同的变量 $\\theta$ 来表示权重。\n",
    "\n",
    "逻辑回归\n",
    "$$ h(z) = \\frac{1}{1+\\exp^{-z}}$$\n",
    "$$z = \\theta_0 x_0 + \\theta_1 x_1 + \\theta_2 x_2 + ... \\theta_N x_N$$\n",
    "我们将 'z' 称为 'logits'（对数几率）。"
]

# Cell 31: Cost function and Gradient
translations["### Part 1.2 Cost function and Gradient"] = [
    "### 第 1.2 节：代价函数和梯度\n",
    "\n",
    "逻辑回归使用的代价函数是所有训练样本上对数损失的平均值：\n",
    "\n",
    "$$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^m y^{(i)}\\log (h(z(\\theta)^{(i)})) + (1-y^{(i)})\\log (1-h(z(\\theta)^{(i)}))\\tag{5} $$\n",
    "* $m$ 是训练样本的数量\n",
    "* $y^{(i)}$ 是第 i 个训练样本的真实标签。\n",
    "* $h(z(\\theta)^{(i)})$ 是模型对第 i 个训练样本的预测。\n",
    "\n",
    "单个训练样本的损失函数为\n",
    "$$ Loss = -1 \\times \\left( y^{(i)}\\log (h(z(\\theta)^{(i)})) + (1-y^{(i)})\\log (1-h(z(\\theta)^{(i)})) \\right)$$\n",
    "\n",
    "* 所有的 $h$ 值都在 0 到 1 之间，因此对数将为负数。这就是为什么要在两个损失项之和前乘以 -1 的原因。\n",
    "* 注意，当模型预测为 1（$h(z(\\theta)) = 1$）且标签 $y$ 也为 1 时，该训练样本的损失为 0。\n",
    "* 类似地，当模型预测为 0（$h(z(\\theta)) = 0$）且真实标签也为 0 时，该训练样本的损失为 0。\n",
    "* 然而，当模型预测接近 1（$h(z(\\theta)) = 0.9999$）而标签为 0 时，对数损失的第二项将变成一个很大的负数，然后乘以整体系数 -1 转换为正的损失值。$-1 \\times (1 - 0) \\times log(1 - 0.9999) \\approx 9.2$ 模型预测越接近 1，损失越大。"
]

# Cell 34: Likewise if model predicts close to 0
translations["* Likewise, if the model predicts close to 0"] = [
    "* 同样地，如果模型预测接近 0（$h(z) = 0.0001$）但真实标签为 1，损失函数中的第一项将变成一个很大的数：$-1 \\times log(0.0001) \\approx 9.2$。预测越接近零，损失越大。"
]

# Cell 36: Update the weights
translations["#### Update the weights"] = [
    "#### 更新权重\n",
    "\n",
    "要更新权重向量 $\\theta$，你将应用梯度下降来迭代地改进模型的预测。\n",
    "代价函数 $J$ 对其中一个权重 $\\theta_j$ 的梯度为：\n",
    "\n",
    "$$\\nabla_{\\theta_j}J(\\theta) = \\frac{1}{m} \\sum_{i=1}^m(h^{(i)}-y^{(i)})x_j \\tag{5}$$\n",
    "* 'i' 是遍历所有 'm' 个训练样本的索引。\n",
    "* 'j' 是权重 $\\theta_j$ 的索引，因此 $x_j$ 是与权重 $\\theta_j$ 关联的特征。\n",
    "\n",
    "* 要更新权重 $\\theta_j$，我们通过减去由 $\\alpha$ 决定的梯度的一部分来调整它：\n",
    "$$\\theta_j = \\theta_j - \\alpha \\times \\nabla_{\\theta_j}J(\\theta) $$\n",
    "* 学习率 $\\alpha$ 是我们选择的一个值，用于控制单次更新的幅度。\n"
]

# Cell 38: Instructions gradient descent
translations["## Instructions: Implement gradient descent function"] = [
    "## 说明：实现梯度下降函数\n",
    "* 迭代次数 `num_iters` 是你使用整个训练集的次数。\n",
    "* 每次迭代，你将使用所有训练样本（共有 `m` 个训练样本）以及所有特征来计算代价函数。\n",
    "* 与其一次更新一个权重 $\\theta_i$，我们可以同时更新列向量中的所有权重：\n",
    "$$\\mathbf{\\theta} = \\begin{pmatrix}\n",
    "\\theta_0\n",
    "\\\\\n",
    "\\theta_1\n",
    "\\\\ \n",
    "\\theta_2 \n",
    "\\\\ \n",
    "\\vdots\n",
    "\\\\ \n",
    "\\theta_n\n",
    "\\end{pmatrix}$$\n",
    "* $\\mathbf{\\theta}$ 的维度为 (n+1, 1)，其中 'n' 是特征数量，额外的一个元素是偏置项 $\\theta_0$（注意对应的特征值 $\\mathbf{x_0}$ 为 1）。\n",
    "* 'logits' 'z' 通过特征矩阵 'x' 与权重向量 'theta' 相乘计算得到。$z = \\mathbf{x}\\mathbf{\\theta}$\n",
    "    * $\\mathbf{x}$ 的维度为 (m, n+1)\n",
    "    * $\\mathbf{\\theta}$：维度为 (n+1, 1)\n",
    "    * $\\mathbf{z}$：维度为 (m, 1)\n",
    "* 预测 'h' 通过对 'z' 中的每个元素应用 sigmoid 计算得到：$h(z) = sigmoid(z)$，维度为 (m,1)。\n",
    "* 代价函数 $J$ 通过取向量 'y' 和 'log(h)' 的点积计算。由于 'y' 和 'h' 都是列向量 (m,1)，将左侧向量转置，使行向量与列向量的矩阵乘法执行点积。\n",
    "$$J = \\frac{-1}{m} \\times \\left(\\mathbf{y}^T \\cdot log(\\mathbf{h}) + \\mathbf{(1-y)}^T \\cdot log(\\mathbf{1-h}) \\right)$$\n",
    "* theta 的更新也是向量化的。由于 $\\mathbf{x}$ 的维度为 (m, n+1)，而 $\\mathbf{h}$ 和 $\\mathbf{y}$ 都是 (m, 1)，我们需要转置 $\\mathbf{x}$ 并将其放在左侧以执行矩阵乘法，从而得到我们需要的 (n+1, 1) 结果：\n",
    "$$\\mathbf{\\theta} = \\mathbf{\\theta} - \\frac{\\alpha}{m} \\times \\left( \\mathbf{x}^T \\cdot \\left( \\mathbf{h-y} \\right) \\right)$$"
]

# Cell 40: Hints gradient descent
translations["<details>    \n<summary>\n    <font size=\"3\" color=\"darkgreen\"><b>Hints</b></font>\n</summary>\n<p>\n<ul>\n    <li>use np.dot for matrix multiplication.</li>"] = [
    "<details>    \n",
    "<summary>\n",
    "    <font size=\"3\" color=\"darkgreen\"><b>提示</b></font>\n",
    "</summary>\n",
    "<p>\n",
    "<ul>\n",
    "    <li>使用 np.dot 进行矩阵乘法。</li>\n",
    "    <li>为确保分数 -1/m 是小数值，将分子或分母（或两者）转换类型，如 `float(1)`，或写 `1.` 表示 1 的浮点版本。</li>\n",
    "</ul>\n",
    "</p>\n",
    "\n"
]

# Cell 43: Expected output gradient descent
translations["#### Expected output\n```\nThe cost after training is 0.67094970."] = [
    "#### 预期输出\n",
    "```\n",
    "The cost after training is 0.67094970.\n",
    "The resulting vector of weights is [4.1e-07, 0.00035658, 7.309e-05]\n",
    "```"
]

# Cell 45: Part 2 Extracting features
translations["## Part 2: Extracting the features"] = [
    "## 第 2 部分：提取特征\n",
    "\n",
    "* 给定推文列表，提取特征并将它们存储在一个矩阵中。你将提取两个特征。\n",
    "    * 第一个特征是推文中积极单词的数量。\n",
    "    * 第二个特征是推文中消极单词的数量。\n",
    "* 然后在这些特征上训练你的逻辑回归分类器。\n",
    "* 在验证集上测试分类器。\n",
    "\n",
    "### 说明：实现 extract_features 函数。\n",
    "* 该函数接收单条推文。\n",
    "* 使用导入的 `process_tweet()` 函数处理推文，并保存推文单词列表。\n",
    "* 遍历处理后单词列表中的每个单词\n",
    "    * 对于每个单词，在 `freqs` 字典中查找该单词带有积极 '1' 标签时的计数。（查找键 (word, 1.0)）\n",
    "    * 同样查找该单词与消极标签 '0' 关联时的计数。（查找键 (word, 0.0)。）"
]

# Cell 47: Hints extract_features
translations["<details>    \n<summary>\n    <font size=\"3\" color=\"darkgreen\"><b>Hints</b></font>\n</summary>\n<p>\n<ul>\n    <li>Make sure you handle cases when the (word, label) key is not found in the dictionary. </li>"] = [
    "<details>    \n",
    "<summary>\n",
    "    <font size=\"3\" color=\"darkgreen\"><b>提示</b></font>\n",
    "</summary>\n",
    "<p>\n",
    "<ul>\n",
    "    <li>确保处理 (word, label) 键在字典中找不到的情况。</li>\n",
    "    <li>在网上搜索关于使用 Python 字典 `.get()` 方法的提示。这里有一个<a href=\"https://www.programiz.com/python-programming/methods/dictionary/get\" > 示例 </a></li>\n",
    "</ul>\n",
    "</p>"
]

# Cell 50: Expected output extract_features test1
translations["#### Expected output\n```\n[[1.00e+00 3.02e+03 6.10e+01]]"] = [
    "#### 预期输出\n",
    "```\n",
    "[[1.00e+00 3.02e+03 6.10e+01]]\n",
    "```"
]

# Cell 53: Expected output extract_features test2
translations["#### Expected output\n```\n[[1. 0. 0.]]"] = [
    "#### 预期输出\n",
    "```\n",
    "[[1. 0. 0.]]\n",
    "```"
]

# Cell 55: Part 3 Training
translations["## Part 3: Training Your Model"] = [
    "## 第 3 部分：训练你的模型\n",
    "\n",
    "训练模型：\n",
    "* 将所有训练样本的特征堆叠成矩阵 `X`。\n",
    "* 调用你在上面实现的 `gradientDescent`。\n",
    "\n",
    "这部分内容已经为你提供。请阅读以理解其原理并运行代码单元。"
]

# Cell 58: Expected Output training
translations["**Expected Output**: \n\n```\nThe cost after training is 0.24216529."] = [
    "**预期输出**：\n",
    "\n",
    "```\n",
    "The cost after training is 0.24216529.\n",
    "The resulting vector of weights is [7e-08, 0.0005239, -0.00055517]\n",
    "```"
]

# Cell 60: Part 4 Test
translations["# Part 4: Test your logistic regression"] = [
    "# 第 4 部分：测试你的逻辑回归\n",
    "\n",
    "现在是时候在模型未见过的新输入上测试你的逻辑回归函数了。\n",
    "\n",
    "#### 说明：编写 `predict_tweet`\n",
    "预测一条推文是积极的还是消极的。\n",
    "\n",
    "* 给定一条推文，处理它，然后提取特征。\n",
    "* 将模型学习到的权重应用于特征以获得 logits。\n",
    "* 对 logits 应用 sigmoid 以获得预测（0 到 1 之间的值）。\n",
    "\n",
    "$$y_{pred} = sigmoid(\\mathbf{x} \\cdot \\theta)$$"
]

# Cell 64: Expected Output predict_tweet
translations["**Expected Output**: \n```\nI am happy -> 0.518580"] = [
    "**预期输出**：\n",
    "```\n",
    "I am happy -> 0.518580\n",
    "I am bad -> 0.494339\n",
    "this movie should have been great. -> 0.515331\n",
    "great -> 0.515464\n",
    "great great -> 0.530898\n",
    "great great great -> 0.546273\n",
    "great great great great -> 0.561561\n",
    "```"
]

# Cell 67: Check performance
translations["## Check performance using the test set"] = [
    "## 使用测试集检查性能\n",
    "使用上面的训练集训练模型后，通过在测试集上测试来检查模型在真实、未见过的数据上的表现。\n",
    "\n",
    "#### 说明：实现 `test_logistic_regression`\n",
    "* 给定测试数据和训练好的模型权重，计算逻辑回归模型的准确率。\n",
    "* 使用你的 `predict_tweet()` 函数对测试集中的每条推文进行预测。\n",
    "* 如果预测 > 0.5，将模型的分类 `y_hat` 设为 1，否则设为 0。\n",
    "* 当 `y_hat` 等于 `test_y` 时预测正确。将所有相等的实例求和并除以 `m`。"
]

# Cell 69: Hints test_logistic_regression
translations["<details>    \n<summary>\n    <font size=\"3\" color=\"darkgreen\"><b>Hints</b></font>\n</summary>\n<p>\n<ul>\n    <li>Use np.asarray() to convert a list to a numpy array</li>"] = [
    "<details>    \n",
    "<summary>\n",
    "    <font size=\"3\" color=\"darkgreen\"><b>提示</b></font>\n",
    "</summary>\n",
    "<p>\n",
    "<ul>\n",
    "    <li>使用 np.asarray() 将列表转换为 numpy 数组</li>\n",
    "    <li>使用 np.squeeze() 将 (m,1) 维数组变为 (m,) 数组</li>\n",
    "</ul>\n",
    "</p>"
]

# Cell 72: Expected Output accuracy
translations["#### Expected Output: \n```0.9950```  \nPretty good!"] = [
    "#### 预期输出：\n",
    "```0.9950```  \n",
    "相当不错！"
]

# Cell 74: Part 5 Error Analysis
translations["# Part 5: Error Analysis"] = [
    "# 第 5 部分：错误分析\n",
    "\n",
    "在这一部分，你将看到一些模型分类错误的推文。你认为为什么会发生这些分类错误？具体来说，你的模型会对哪种类型的推文分类错误？"
]

# Cell 77: Later in specialization
translations["Later in this specialization, we will see how we can use deep learning to improve the prediction performance."] = [
    "在本专项课程的后续内容中，我们将学习如何使用深度学习来提高预测性能。"
]

# Cell 78: Part 6
translations["# Part 6: Predict with your own tweet"] = [
    "# 第 6 部分：用你自己的推文进行预测"
]

# 遍历单元格，替换 markdown 内容
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        # 尝试匹配翻译
        matched = False
        for key, translated in translations.items():
            if source_text.startswith(key) or key in source_text[:100]:
                cell['source'] = translated
                matched = True
                break
        if not matched:
            # 打印未匹配的以便检查
            print(f"未匹配的 markdown 单元格（前80字符）: {source_text[:80]}")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"翻译完成，已保存到: {output_path}")
