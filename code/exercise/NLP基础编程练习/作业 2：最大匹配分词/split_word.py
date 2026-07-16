def lode_file(file_path:str) -> list:
    with open(file_path,'r') as fp:
        lines = [line.strip() for line in fp.readlines()]
    return lines


def get_dic_max_len(word_list:list) -> int:
    max_len = 0
    for word in word_list:
        if word:
            max_len = max(max_len, len(word))
    return max_len

def split_word(text:str, word_list:list) -> list:
    max_len = get_dic_max_len(word_list)
    



