from collections import defaultdict, Counter


def load_conll(file_path: str) -> dict:
    """Load a CoNLL file for HMM POS tagging."""
    tokens = []
    sentences = []
    current_sentence = {
        "words": [],
        "tags": [],
        "tokens": [],
    }

    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()

            # Empty line means the end of one sentence.
            if not line:
                if current_sentence["tokens"]:
                    sentences.append(current_sentence)
                    current_sentence = {
                        "words": [],
                        "tags": [],
                        "tokens": [],
                    }
                continue

            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(f"Invalid CoNLL line: {line}")

            token = {
                "id": int(columns[0]),
                "word": columns[1],
                "pos": columns[3],
                "head": int(columns[6]),
                "dep": columns[7],
            }

            tokens.append(token)
            current_sentence["tokens"].append(token)
            current_sentence["words"].append(token["word"])
            current_sentence["tags"].append(token["pos"])

    if current_sentence["tokens"]:
        sentences.append(current_sentence)

    return {
        "sentences": sentences,
        "tokens": tokens,
    }

def train_hmm(train_data: dict):
    """Train HMM model."""
    # 初始化模型参数
    start_count = Counter()
    tag_counts = Counter()
    transition_count = defaultdict(Counter)
    emission_count = defaultdict(Counter)

    vocab = set()
    tag_set = set()
    for sentence in train_data["sentences"]:
        words = sentence["words"]
        tags = sentence["tags"]
        
        if not words:
            continue
        
        start_count[tags[0]] += 1
        
        for i in range(len(words)):
            word = words[i]
            tag = tags[i]
            
            vocab.add(word)
            tag_set.add(tag)
            tag_counts[tag] += 1

            emission_count[tag][word] += 1

            if i > 0:
                transition_count[tags[i-1]][tag] += 1
        
    return start_count, tag_counts, transition_count, emission_count, vocab, tag_set



if __name__ == "__main__":
    train_data = load_conll("data/train.conll")
    print(train_data["sentences"][0]["words"])
    print(train_data["sentences"][0]["tags"])
    print(train_data)
