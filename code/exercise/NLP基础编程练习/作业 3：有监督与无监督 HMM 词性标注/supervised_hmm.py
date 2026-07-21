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


if __name__ == "__main__":
    train_data = load_conll("data/train.conll")
    print(train_data["sentences"][0]["words"])
    print(train_data["sentences"][0]["tags"])
    print(train_data)
