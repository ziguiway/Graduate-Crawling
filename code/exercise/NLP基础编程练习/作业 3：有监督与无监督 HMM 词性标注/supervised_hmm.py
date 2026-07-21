def load_conll(file_path: str) -> dict:
    data = []
    with open(file_path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = line.split("\t")
        tmp = {
            id: line[0],
            word: line[1],
            pos: line[2],
            tag: line[3],
        }
        data.append(tmp)
    
    return data

load_conll("data/train.conll")