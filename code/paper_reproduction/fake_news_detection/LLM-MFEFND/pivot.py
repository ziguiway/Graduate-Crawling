import torch
import torch.nn as nn


class MLP_trans(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        head_num=4,
        dropout=0.1,
        attention_dropout=0.0,
        initializer_range=0.02,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            head_num,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.apply(lambda module: self._init_weights(module, initializer_range))

    @staticmethod
    def _init_weights(module, initializer_range):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights
