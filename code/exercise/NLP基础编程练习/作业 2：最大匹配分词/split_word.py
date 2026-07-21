def lode_file(file_path: str) -> list:
    with open(file_path, 'r') as fp:
        lines = [line.strip("\n") for line in fp.readlines()]
    return lines


def get_dic_max_len(word_list: list) -> int:
    max_len = 0
    for word in word_list:
        if word:
            max_len = max(max_len, len(word))
    return max_len


def split_word(word_list: list, text: str) -> list:
    result = []
    max_len = get_dic_max_len(word_list)
    i = 0
    while i < len(text):
        j = max_len
        while j > 0:
            word = text[i:i + j]
            if word in word_list:
                result.append(word)
                i = i + j
                break
            if j == 1:
                result.append(text[i])
                i += 1
                break
            j -= 1
    return result


def identify_word(text: str, answer: str) -> int:
    """位置对齐版（有缺陷，仅作对比参考）：
    逐位置对比 text 和 answer 的词列表，相等的计数。
    问题：一旦某处切分不一致，后面所有词位置错位，会低估正确数。
    """
    split_text = text.split()
    split_answer = answer.split()
    count = 0
    for a, b in zip(split_text, split_answer):
        if a == b:
            count += 1
    return count


def boundaries(words: list) -> set:
    """把词列表转成切分边界位置集合。
    例：['中国','人民'] → 累积字符位置 {2, 4}
    """
    pos = 0
    b = set()
    for w in words:
        pos += len(w)
        b.add(pos)
    return b


def evaluate(my_words_per_line: list, answer_words_per_line: list) -> tuple:
    """字符边界对齐版：对比切分边界位置，重合的边界数 = 正确切分数。
    输入是按行切好的词列表：[[w1,w2,...], [w1,w2,...], ...]
    返回 (correct, total_my, total_ans)。
    """
    correct = 0
    total_my = 0
    total_ans = 0
    for my_line, ans_line in zip(my_words_per_line, answer_words_per_line):
        # 边界重合 = 正确切分
        correct += len(boundaries(my_line) & boundaries(ans_line))
        total_my += len(my_line)
        total_ans += len(ans_line)
    return correct, total_my, total_ans


def main():
    word_list = lode_file("作业 2：最大匹配分词/data/Dict.txt")
    text = "\n".join(lode_file("作业 2：最大匹配分词/data/Sentence.txt"))
    result = split_word(word_list, text)
    out = ""
    for word in result:
        if word == '\n':
            out += word
            continue
        out += word + " "

    with open("作业 2：最大匹配分词/data/MyOut.txt", 'w') as fp:
        fp.write(out)

    print("切出的总词数:", len([w for w in result if w != '\n']))

    # 把 result（扁平词列表）按 '\n' 分行
    my_words_per_line = []
    cur = []
    for w in result:
        if w == '\n':
            my_words_per_line.append(cur)
            cur = []
        else:
            cur.append(w)
    my_words_per_line.append(cur)   # 最后一行

    # 答案按行加载并切词
    answer_lines = lode_file("作业 2：最大匹配分词/data/Answer.txt")
    answer_words_per_line = [line.split() for line in answer_lines]

    # 评价
    correct, total_my, total_ans = evaluate(my_words_per_line, answer_words_per_line)
    p = correct / total_my if total_my else 0
    r = correct / total_ans if total_ans else 0
    f = 2 * p * r / (p + r) if (p + r) else 0

    print()
    print("=== 评价结果（字符边界对齐版）===")
    print(f"正确识别的词数: {correct}")
    print(f"识别出的总体个数: {total_my}")
    print(f"测试集中的总体个数: {total_ans}")
    print(f"精确率 P: {p:.5f}")
    print(f"召回率 R: {r:.5f}")
    print(f"F 值:     {f:.5f}")

    print()
    print("=== 老师给的参考答案 ===")
    print("正确识别的词数: 20263")
    print("识别出的总体个数: 20397")
    print("测试集中的总体个数: 20454")
    print("精确率 P: 0.99343")
    print("召回率 R: 0.99066")
    print("F 值:     0.99204")


if __name__ == "__main__":
    main()