# SPDX-License-Identifier: GPL-3.0-or-later

# https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

import math
from typing import Optional

D_MODEL_DEFAULT = 512
DIM_FEEDFORWARD_DEFAULT = 2048


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Requires nested tensors.
    """

    def __init__(
        self,
        embed_dim,
        key_dim,
        value_dim,
        out_dim,
        num_heads,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # E
        self.key_dim = key_dim  # E_k
        self.value_dim = value_dim  # E_v
        self.out_dim = out_dim
        self.num_heads = num_heads  # H
        self.dropout = dropout_prob
        self.query_proj = nn.Linear(embed_dim, num_heads * key_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, num_heads * key_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, num_heads * value_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * value_dim, out_dim, bias=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        heads = F.scaled_dot_product_attention(
            query=query.unflatten(-1, (self.num_heads, self.key_dim)).transpose(-2, -3),
            key=key.unflatten(-1, (self.num_heads, self.key_dim)).transpose(-2, -3),
            value=value.unflatten(-1, (self.num_heads, self.value_dim)).transpose(
                -2, -3
            ),
            scale=1 / math.sqrt(self.key_dim),
            is_causal=is_causal,
            dropout_p=(self.dropout if self.training else 0.0),
        )  # (N, H, L, E_v)
        concat_heads = heads.transpose(-2, -3).flatten(-2, -1)  # (N, L, H * E_v)
        return self.out_proj(concat_heads)  # (N, L, E)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dim_feedforward: int):
        super().__init__()
        self.d_model = embed_dim
        self.dim_feedforward = dim_feedforward
        self.sequential = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.d_model),
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.sequential(X)


# TODO: check & complete regularization
class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob

        self.multi_head = MultiHeadAttention(
            embed_dim=embed_dim,
            key_dim=embed_dim / num_heads,
            value_dim=embed_dim / num_heads,
            out_dim=embed_dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
        )
        self.ffn = FeedForward(embed_dim, dim_feedforward)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X: Tensor) -> Tensor:
        for _ in range(self.num_layers):
            X = self.layer_norm(
                X + self.dropout(self.multi_head(X, X, X, is_causal=False))
            )
            X = self.layer_norm(X + self.dropout(self.ffn(X)))
        return X


# TODO: check & complete regularization
class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers

        self.multi_head_1 = MultiHeadAttention(
            embed_dim=embed_dim,
            key_dim=embed_dim / num_heads,
            value_dim=embed_dim / num_heads,
            out_dim=embed_dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
        )
        self.multi_head_2 = MultiHeadAttention(
            embed_dim=embed_dim,
            key_dim=embed_dim / num_heads,
            value_dim=embed_dim / num_heads,
            out_dim=embed_dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
        )
        self.ffn = FeedForward(embed_dim, dim_feedforward)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X, encoder_out):
        for _ in range(self.num_layers):
            mha = self.multi_head_1(X, X, X, is_causal=True)
            mha = self.dropout(mha)
            X = self.layer_norm(X + mha)

            mha = self.multi_head_2(encoder_out, encoder_out, X, is_causal=False)
            mha = self.dropout(mha)
            X = self.layer_norm(X + mha)

            ffn = self.ffn(X)
            X = self.layer_norm(X + ffn)

        return X


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        # self.max_len = max_len
        self.vocab_size = vocab_size
        # self.d_model = embed_dim
        self.dim_feedforward = dim_feedforward
        self.input_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.output_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        # self.PE = self._generate_positional_encoding()  # PE (Positional Encoding)

        self.encoder = Encoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers,
            dropout_prob=dropout_prob,
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_decoder_layers,
            dropout_prob=dropout_prob,
        )

    # def _generate_positional_encoding(self):
    #     pos = torch.arange(0, self.max_len, step=1).unsqueeze(1)
    #     two_i = torch.arange(0, self.d_model, step=2)
    #     pe = torch.zeros((self.max_len, self.d_model))
    #     angles = pos / torch.pow(10_000, two_i / self.d_model)
    #     pe[:, 0::2] = torch.sin(angles)
    #     pe[:, 1::2] = torch.cos(angles if self.d_model % 2 == 0 else angles[:, 0:-1])
    #     return pe

    def forward(self, input):
        return input  # TODO: fix
