def lode_file(file_path:str) -> list:
    with open(file_path,'r') as fp:
        lines = [line.strip("\n") for line in fp.readlines()]
    return lines


def get_dic_max_len(word_list:list) -> int:
    max_len = 0
    for word in word_list:
        if word:
            max_len = max(max_len, len(word))
    return max_len

def split_word(word_list:list, text:str) -> list:
    result = []
    max_len = get_dic_max_len(word_list)
    i = 0
    while i < len(text):
        j = max_len
        while j > 0:
            word = text[i:i+j]
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
              


        
    

def main():
    word_list = lode_file("作业 2：最大匹配分词/data/Dict.txt")
    text = "/n".join(lode_file("作业 2：最大匹配分词/data/Sentence.txt"))
    print(text)
    result = split_word(word_list, text)
    print(" ".join(result).strip())
    with open("作业 2：最大匹配分词/data/MyOut.txt",'w') as fp:
        fp.write(" ".join(result).strip())

    print(len(result))

    



if __name__ == "__main__":
    main()