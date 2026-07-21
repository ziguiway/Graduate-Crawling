import math
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

def train_hmm(train_data: dict, alpha=1.0):
    """Train HMM model."""
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

    return {
        "start_count": start_count,
        "tag_counts": tag_counts,
        "transition_count": transition_count,
        "emission_count": emission_count,
        "vocab": vocab,
        "tag_set": tag_set,
        "sentence_count": len(train_data["sentences"]),
        "alpha": alpha,
    }


def get_start_prob(tag: str, model: dict) -> float:
    """P(tag appears at the beginning of a sentence)."""
    return model["start_count"][tag] / model["sentence_count"]


def get_transition_prob(prev_tag: str, tag: str, model: dict) -> float:
    """P(tag | prev_tag)."""
    if model["tag_counts"][prev_tag] == 0:
        return 0.0

    return model["transition_count"][prev_tag][tag] / model["tag_counts"][prev_tag]


def get_emission_prob(tag: str, word: str, model: dict) -> float:
    """P(word | tag), estimated with add-alpha smoothing."""
    alpha = model["alpha"]
    vocab_size = len(model["vocab"])

    return (
        model["emission_count"][tag][word] + alpha
    ) / (
        model["tag_counts"][tag] + alpha * vocab_size
    )

def safe_log(prob: float) -> float:
    """Return log(prob), using -inf for impossible paths."""
    return math.log(prob) if prob > 0.0 else -math.inf


def viterbi(words: list[str], model: dict) -> list[str]:
    """Find the best POS tag sequence for one sentence with Viterbi."""
    if not words:
        return []

    tags = sorted(model["tag_set"])
    dp = [{} for _ in words]
    backpointer = [{} for _ in words]

    # 1. Initialization: START -> tag, then tag emits the first word.
    for tag in tags:
        dp[0][tag] = (
            safe_log(get_start_prob(tag, model))
            + safe_log(get_emission_prob(tag, words[0], model))
        )
        backpointer[0][tag] = None

    # 2. Recursion: for each current tag, choose the best previous tag.
    for i in range(1, len(words)):
        word = words[i]

        for tag in tags:
            best_prev_tag = None
            best_score = -math.inf

            for prev_tag in tags:
                score = (
                    dp[i - 1][prev_tag]
                    + safe_log(get_transition_prob(prev_tag, tag, model))
                    + safe_log(get_emission_prob(tag, word, model))
                )

                if score > best_score:
                    best_score = score
                    best_prev_tag = prev_tag

            dp[i][tag] = best_score
            backpointer[i][tag] = best_prev_tag

    # 3. Termination without STOP: choose the best tag at the last word.
    best_last_tag = max(tags, key=lambda tag: dp[-1][tag])

    # 4. Backtracking: recover the best path from right to left.
    best_tags = [best_last_tag]
    for i in range(len(words) - 1, 0, -1):
        best_tags.append(backpointer[i][best_tags[-1]])

    best_tags.reverse()
    return best_tags

if __name__ == "__main__":
    train_data = load_conll("data/train.conll")
    model = train_hmm(train_data)

    print("第一句话 words:", train_data["sentences"][0]["words"])
    print("第一句话 tags:", train_data["sentences"][0]["tags"])
    print("词表大小:", len(model["vocab"]))
    print("词性数量:", len(model["tag_set"]))
    print("P(NR at start) =", get_start_prob("NR", model))
    print("P(VV | NR) =", get_transition_prob("NR", "VV", model))
    print("P(中国 | NR) =", get_emission_prob("NR", "中国", model))
    print("P(不存在的词 | NR) =", get_emission_prob("NR", "不存在的词", model))

    words = train_data["sentences"][0]["words"]
    print("第一句话 Viterbi 预测 tags:", viterbi(words, model))
