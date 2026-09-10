"""按照课件流程，使用普通 Perceptron 实现局部线性模型词性标注。

代码明确实现课件中的：

1. 收集训练集中的完整特征，建立特征空间 E；
2. 将特征字符串映射为整数编号；
3. 用激活的特征编号表示稀疏特征向量 f(S, i, t)；
4. 用列表表示权重向量 w；
5. 通过 w 与 f(S, i, t) 的稀疏点乘计算分数；
6. 使用普通 Perceptron 更新权重。

暂时不实现 partial feature、Averaged Perceptron 和累积权重 v 的延迟更新。
"""

import argparse
from pathlib import Path


Sentence = dict[str, list[str]]
FeatureSpace = dict[str, int]
Weights = list[float]


def load_conll(file_path: Path) -> list[Sentence]:
    """读取 CoNLL 文件，只保留词语和词性两列。"""
    sentences: list[Sentence] = []
    words: list[str] = []
    tags: list[str] = []

    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                if words:
                    sentences.append({"words": words, "tags": tags})
                    words = []
                    tags = []
                continue

            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(f"不合法的 CoNLL 行：{line}")

            words.append(columns[1])
            tags.append(columns[3])

    if words:
        sentences.append({"words": words, "tags": tags})

    return sentences


def extract_feature_strings(
    words: list[str],
    index: int,
    tag: str,
) -> list[str]:
    """按照课件表 1，为位置 index 和候选词性 tag 构造完整字符串特征。

    这里故意把 tag 拼进每一个特征。以后实现 partial feature 时，才会把
    与 tag 无关的部分单独抽出来，并通过 offset 得到完整特征编号。
    """
    word = words[index]
    prev_word = words[index - 1] if index > 0 else "<BOS>"
    next_word = words[index + 1] if index + 1 < len(words) else "<EOS>"

    prev_last_char = words[index - 1][-1] if index > 0 else "<BOS_CHAR>"
    next_first_char = (
        words[index + 1][0]
        if index + 1 < len(words)
        else "<EOS_CHAR>"
    )
    first_char = word[0]
    last_char = word[-1]

    features = [
        # 02: t + wi
        f"02|tag={tag}|word={word}",
        # 03: t + wi-1
        f"03|tag={tag}|prev_word={prev_word}",
        # 04: t + wi+1
        f"04|tag={tag}|next_word={next_word}",
        # 05: t + wi + ci-1,-1
        f"05|tag={tag}|word={word}|prev_last={prev_last_char}",
        # 06: t + wi + ci+1,0
        f"06|tag={tag}|word={word}|next_first={next_first_char}",
        # 07: t + ci,0
        f"07|tag={tag}|first_char={first_char}",
        # 08: t + ci,-1
        f"08|tag={tag}|last_char={last_char}",
    ]

    # 09、10、11：当前词内部的汉字，不包括首字和尾字。
    for char in word[1:-1]:
        features.append(f"09|tag={tag}|inner_char={char}")
        features.append(
            f"10|tag={tag}|first_char={first_char}|inner_char={char}"
        )
        features.append(
            f"11|tag={tag}|last_char={last_char}|inner_char={char}"
        )

    # 12：当前词只有一个汉字时，同时观察前一个词末字和后一个词首字。
    if len(word) == 1:
        features.append(
            f"12|tag={tag}|word={word}"
            f"|prev_last={prev_last_char}|next_first={next_first_char}"
        )

    # 13：当前词包含连续重复的汉字。
    for char_index in range(len(word) - 1):
        if word[char_index] == word[char_index + 1]:
            features.append(
                f"13|tag={tag}|char={word[char_index]}|consecutive"
            )

    # 14、15：长度为 1 到 4 的前缀和后缀。
    max_affix_length = min(4, len(word))
    for length in range(1, max_affix_length + 1):
        features.append(
            f"14|tag={tag}|length={length}|prefix={word[:length]}"
        )
        features.append(
            f"15|tag={tag}|length={length}|suffix={word[-length:]}"
        )

    # 课件中的特征向量是 0/1 向量。同一个特征即使由多个位置重复触发，
    # 在向量中也只能取值 1，所以这里按出现顺序去重。
    return list(dict.fromkeys(features))


def build_feature_space(sentences: list[Sentence]) -> FeatureSpace:
    """扫描训练集，建立课件中的特征空间 E。

    每个训练实例都带有人工标注的正确词性，因此使用 gold tag 实例化
    特征模板。每个不同的完整特征字符串对应特征空间中的唯一整数下标。
    """
    feature_to_id: FeatureSpace = {}

    for sentence in sentences:
        words = sentence["words"]
        gold_tags = sentence["tags"]

        for index, gold_tag in enumerate(gold_tags):
            feature_strings = extract_feature_strings(
                words, index, gold_tag
            )

            for feature in feature_strings:
                if feature not in feature_to_id:
                    feature_to_id[feature] = len(feature_to_id)

    return feature_to_id


def vectorize_features(
    words: list[str],
    index: int,
    tag: str,
    feature_to_id: FeatureSpace,
) -> list[int]:
    """用激活特征的编号表示稀疏特征向量 f(S, i, t)。

    完整向量的维度是 |E|，但绝大多数位置都是 0。这里不保存所有的 0，
    只返回值为 1 的位置，也就是被激活的特征编号。
    """
    feature_strings = extract_feature_strings(words, index, tag)

    return [
        feature_to_id[feature]
        for feature in feature_strings
        if feature in feature_to_id
    ]


def dot_product(feature_ids: list[int], weights: Weights) -> float:
    """计算 w 与稀疏特征向量 f(S, i, t) 的点积。

    feature_ids 中的每个位置在 f 中都取值 1，未出现的位置都取值 0，
    因此点积等价于把所有激活位置对应的 w 值相加。
    """
    return sum(weights[feature_id] for feature_id in feature_ids)


def predict_tag(
    words: list[str],
    index: int,
    tag_set: list[str],
    feature_to_id: FeatureSpace,
    weights: Weights,
) -> tuple[str, list[int]]:
    """根据 argmax_t w · f(S, i, t) 返回分数最高的词性。"""
    best_tag = tag_set[0]
    best_feature_ids = vectorize_features(
        words, index, best_tag, feature_to_id
    )
    best_score = dot_product(best_feature_ids, weights)

    for tag in tag_set[1:]:
        # 暂未实现 partial feature：每换一个候选词性，都重新抽取完整特征。
        feature_ids = vectorize_features(
            words, index, tag, feature_to_id
        )
        score = dot_product(feature_ids, weights)

        if score > best_score:
            best_tag = tag
            best_feature_ids = feature_ids
            best_score = score

    return best_tag, best_feature_ids


def perceptron_update(
    gold_feature_ids: list[int],
    predicted_feature_ids: list[int],
    weights: Weights,
) -> None:
    """执行 w = w + f(S, i, gold) - f(S, i, predicted)。"""
    for feature_id in gold_feature_ids:
        weights[feature_id] += 1.0

    for feature_id in predicted_feature_ids:
        weights[feature_id] -= 1.0


def train_one_epoch(
    sentences: list[Sentence],
    tag_set: list[str],
    feature_to_id: FeatureSpace,
    weights: Weights,
) -> tuple[int, int]:
    """顺序遍历一次训练集，并返回预测错误数和总词数。"""
    mistakes = 0
    total = 0

    for sentence in sentences:
        words = sentence["words"]
        gold_tags = sentence["tags"]

        for index, gold_tag in enumerate(gold_tags):
            predicted_tag, predicted_feature_ids = predict_tag(
                words, index, tag_set, feature_to_id, weights
            )
            total += 1

            if predicted_tag == gold_tag:
                continue

            mistakes += 1
            gold_feature_ids = vectorize_features(
                words, index, gold_tag, feature_to_id
            )
            perceptron_update(
                gold_feature_ids,
                predicted_feature_ids,
                weights,
            )

    return mistakes, total


def evaluate(
    sentences: list[Sentence],
    tag_set: list[str],
    feature_to_id: FeatureSpace,
    weights: Weights,
) -> tuple[int, int, float]:
    """计算逐词词性标注准确率。"""
    correct = 0
    total = 0

    for sentence in sentences:
        words = sentence["words"]
        gold_tags = sentence["tags"]

        for index, gold_tag in enumerate(gold_tags):
            predicted_tag, _ = predict_tag(
                words,
                index,
                tag_set,
                feature_to_id,
                weights,
            )
            correct += int(predicted_tag == gold_tag)
            total += 1

    accuracy = correct / total if total else 0.0
    return correct, total, accuracy


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    shared_data_dir = script_dir.parent / "data"

    parser = argparse.ArgumentParser(
        description="使用普通 Perceptron 训练局部线性词性标注模型。"
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=shared_data_dir / "train.conll",
        help="训练集路径。",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=shared_data_dir / "dev.conll",
        help="开发集路径。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="训练轮数，默认 5。",
    )
    parser.add_argument(
        "--max-train-sentences",
        type=int,
        default=None,
        help="只使用前 N 个训练句子，便于快速观察代码流程。",
    )
    parser.add_argument(
        "--max-dev-sentences",
        type=int,
        default=None,
        help="只评价前 N 个开发句子，便于快速验证。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_sentences = load_conll(args.train)
    dev_sentences = load_conll(args.dev)

    if args.max_train_sentences is not None:
        train_sentences = train_sentences[: args.max_train_sentences]
    if args.max_dev_sentences is not None:
        dev_sentences = dev_sentences[: args.max_dev_sentences]

    tag_set = sorted(
        {
            tag
            for sentence in train_sentences
            for tag in sentence["tags"]
        }
    )
    if not tag_set:
        raise ValueError("训练集中没有词性。")

    feature_to_id = build_feature_space(train_sentences)
    weights: Weights = [0.0] * len(feature_to_id)

    print(f"训练句子数：{len(train_sentences)}")
    print(f"开发句子数：{len(dev_sentences)}")
    print(f"词性数量：{len(tag_set)}")
    print(f"特征空间维度 |E|：{len(feature_to_id)}")
    print(f"词性集合：{tag_set}")

    first_words = train_sentences[0]["words"]
    first_tag = train_sentences[0]["tags"][0]
    print("\n第一个词：", first_words[0])
    print("正确词性：", first_tag)
    print("它触发的完整特征及其在 E 中的编号：")
    for feature in extract_feature_strings(first_words, 0, first_tag):
        print(f"  {feature_to_id[feature]:>6}  {feature}")

    for epoch in range(1, args.epochs + 1):
        mistakes, train_total = train_one_epoch(
            train_sentences,
            tag_set,
            feature_to_id,
            weights,
        )
        correct, dev_total, dev_accuracy = evaluate(
            dev_sentences,
            tag_set,
            feature_to_id,
            weights,
        )

        print(
            f"\nEpoch {epoch}: "
            f"训练错误 {mistakes}/{train_total}，"
            f"开发集准确率 {correct}/{dev_total} = {dev_accuracy:.4%}，"
            f"非零权重数 {sum(weight != 0 for weight in weights)}"
        )


if __name__ == "__main__":
    main()
