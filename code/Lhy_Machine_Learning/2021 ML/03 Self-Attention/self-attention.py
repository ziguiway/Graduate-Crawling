import numpy as np


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def self_attention(X,W_Q,W_K,W_V):
    Q = W_Q @ X
    K = W_K @ X
    V = W_V @ X

    attention_scroce = K.T @ Q

    attention_scroce = softmax(attention_scroce)

    output = attention_scroce @ V

    return output

