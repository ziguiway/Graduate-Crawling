import numpy as np


def self_attention(X,W_Q,W_K,W_V):
    Q = W_Q @ X
    K = W_k @ X
    V = W_V @ X

    attention_scroce = K.T @ Q
    
     