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


class BidirectionalCrossAttention(nn.Module):
    """News-auxiliary interaction from Eqs. (6)-(13)."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 4):
        super().__init__()
        self.aux_to_news = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.news_to_aux = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=1)
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        news: torch.Tensor,
        auxiliary: torch.Tensor,
        news_mask: torch.Tensor | None = None,
        auxiliary_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        news_padding = news_mask == 0 if news_mask is not None else None
        auxiliary_padding = auxiliary_mask == 0 if auxiliary_mask is not None else None

        aux_attended_news, _ = self.aux_to_news(
            query=auxiliary,
            key=news,
            value=news,
            key_padding_mask=news_padding,
            need_weights=False,
        )
        news_attended_auxiliary, _ = self.news_to_aux(
            query=news,
            key=auxiliary,
            value=auxiliary,
            key_padding_mask=auxiliary_padding,
            need_weights=False,
        )
        forward_feature = self._masked_mean(aux_attended_news, auxiliary_mask)
        reverse_feature = self._masked_mean(news_attended_auxiliary, news_mask)
        weight = self.score(reverse_feature)
        interaction = weight * forward_feature
        details = {
            "forward_feature": forward_feature,
            "reverse_feature": reverse_feature,
            "weight": weight,
        }
        return interaction, weight, details


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


class HierarchicalProgressiveTransformer(nn.Module):
    """HPT from Eqs. (14)-(32) of the LLM-MFEFND paper.

    Each input feature is projected into ``num_tokens`` feature tokens.  A
    shared sequence R is initialized from all feature sequences and then
    updated in the paper's order.  Each update keeps half of the previous R,
    matching R_next = (R_candidate + R_previous) / 2.
    """

    FEATURE_NAMES = ("text", "image", "aligned", "background", "comments")

    def __init__(
        self,
        feature_dim: int = 768,
        num_tokens: int = 5,
        num_heads: int = 4,
        num_rounds: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if num_tokens < 1 or num_rounds < 1:
            raise ValueError("num_tokens and num_rounds must be positive")

        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.num_rounds = num_rounds
        self.feature_projections = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [MLP_trans(feature_dim, feature_dim, dropout=dropout) for _ in range(num_tokens)]
                )
                for name in self.FEATURE_NAMES
            }
        )
        self.fusion_layers = nn.ModuleList(
            [
                TransformerLayer(
                    feature_dim,
                    head_num=num_heads,
                    dropout=dropout,
                    attention_dropout=0.0,
                )
                for _ in self.FEATURE_NAMES
            ]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(num_tokens * feature_dim, feature_dim),
            nn.ReLU(),
        )

    def _project_feature(self, name: str, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 2 or feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"{name} must have shape [batch, {self.feature_dim}], got {tuple(feature.shape)}"
            )
        return torch.stack([projection(feature) for projection in self.feature_projections[name]], dim=1)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        aligned: torch.Tensor,
        background: torch.Tensor,
        comments: torch.Tensor,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        inputs = {
            "text": text,
            "image": image,
            "aligned": aligned,
            "background": background,
            "comments": comments,
        }
        feature_states = {
            name: self._project_feature(name, feature)
            for name, feature in inputs.items()
        }
        shared = sum(feature_states.values()) / len(feature_states)
        trace: dict[str, object] = {
            "order": list(self.FEATURE_NAMES),
            "shared_shapes": [tuple(shared.shape)],
            "rounds": [],
        }

        for round_idx in range(self.num_rounds):
            round_trace = []
            for stage_idx, name in enumerate(self.FEATURE_NAMES):
                stage_input = torch.cat([feature_states[name], shared], dim=1)
                stage_output, _ = self.fusion_layers[stage_idx](stage_input)
                feature_states[name] = stage_output[:, : self.num_tokens]
                candidate = stage_output[:, self.num_tokens :]
                shared = (candidate + shared) / 2.0
                round_trace.append(
                    {
                        "feature": name,
                        "shared_shape": tuple(shared.shape),
                    }
                )
            trace["rounds"].append(round_trace)
            trace["shared_shapes"].append(tuple(shared.shape))

        fused = self.output_projection(shared.reshape(shared.size(0), -1))
        if return_trace:
            return fused, trace
        return fused
