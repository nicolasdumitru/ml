# SPDX-License-Identifier: GPL-3.0-or-later

# https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

D_MODEL_DEFAULT = 512
DIM_FEEDFORWARD_DEFAULT = 2048


# TODO: Multi-Head Attention mask
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_MODEL_DEFAULT, heads=8, d_k=64, d_v=64):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.query_projector = nn.Linear(d_model, heads * d_k, bias=False)
        self.key_projector = nn.Linear(d_model, heads * d_k, bias=False)
        self.query_projector = nn.Linear(d_model, heads * d_v, bias=False)
        self.output_projector = nn.Linear(heads * d_v, d_model, bias=False)

    def attention(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        d_k = torch.tensor(self.d_k)
        return F.softmax(Q @ K.mT / torch.sqrt(d_k), dim=-1) @ V

    def forward(self, X):
        Q = self.query_projector(X)
        K = self.key_projector(X)
        V = self.query_projector(X)
        concat_heads = self.attention(Q, K, V)
        return self.output_projector(concat_heads)


class FeedForward(nn.Module):
    def __init__(
        self, d_model=D_MODEL_DEFAULT, dim_feedforward=DIM_FEEDFORWARD_DEFAULT
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.sequential = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.d_model),
        )

    def forward(self, X):
        return self.sequential(X)


class Encoder(nn.Module):
    # TODO: parameterize
    def __init__(
        self,
        sequence_length,
        stack_size=6,
        d_model=D_MODEL_DEFAULT,
        dim_feedforward=DIM_FEEDFORWARD_DEFAULT,
    ):
        super().__init__()
        self.stack_size = stack_size
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.multi_head = MultiHeadAttention(self.d_model, heads=8, d_k=8, d_v=64)
        self.ffn = FeedForward(self.d_model, self.dim_feedforward)

    def forward(self, X):
        for _ in range(self.stack_size):
            X = F.layer_norm(X + self.multi_head(X))
            X = F.layer_norm(X + self.ffn(X))
        return X


class Decoder(nn.Module):
    def __init__(self, sequence_length, stack_size=6):
        super().__init__()
        self.stack_size = stack_size

    def forward(self, input):
        return input  # TODO: fix


class Transformer(nn.Module):
    def __init__(
        self,
        max_len,
        vocab_size,
        d_model=D_MODEL_DEFAULT,
        dim_feedforward=DIM_FEEDFORWARD_DEFAULT,
        encoder_size=6,
        decoder_size=6,
    ):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.input_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.output_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

        self.PE = self._generate_positional_encoding()  # PE (Positional Encoding)

        self.encoder = Encoder(encoder_size, self.dim_feedforward)
        self.decoder = Decoder(decoder_size)

    def _generate_positional_encoding(self):
        pos = torch.arange(0, self.max_len, step=1).unsqueeze(1)
        two_i = torch.arange(0, self.d_model, step=2)
        pe = torch.zeros((self.max_len, self.d_model))
        angles = pos / torch.pow(10_000, two_i / self.d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles if self.d_model % 2 == 0 else angles[:, 0:-1])
        return pe

    def forward(self, input):
        return input  # TODO: fix
