def get_dic_max_len():
    with open('作业 2：最大匹配分词/data/Dict.txt','r') as fp:
        max_len = 0
        for line in fp:
            line = line.strip()
            if line:
                max_len = max(max_len, len(line))
    return max_len

print(get_dic_max_len())