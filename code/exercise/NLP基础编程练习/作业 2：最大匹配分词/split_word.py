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
    
    while True:
        if not text:
            break
        tmp = text[:max_len]





def main():
    word_list = lode_file("")
    text = "我来到北京清华大学"
    result = split_word(word_list, text)
    print(result)