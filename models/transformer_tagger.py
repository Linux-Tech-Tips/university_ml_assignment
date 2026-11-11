# Second Candidate Model - Encoder-only Transformer

import torch
from torch import nn

from model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerTagger(Model):

    def __init__(self):
        pass

    def forward_pass(self, inputs: list) -> list:
        # TODO Implement
        pass

class Transformer(nn.Module):
    # TODO Implement nn.Module built out of N transformer encoder blocks, ensure proper weight init, forward pass and backprop
    pass

class _TransformerEncoderBlock(nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout_p: float = 0.5):
        super().__init__()
        # Multi-headed self-attention component
        self.attn = _MultiHeadAttention(embedding_dim, num_heads, dropout_p)
        self.attn_norm = nn.LayerNorm(embedding_dim)
        # feed-forward component - up projection -> GELU -> down projection
        self.up_proj = nn.Linear(embedding_dim, ff_dim)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.ff_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.tensor, pad_mask: torch.tensor = None):
        # attention component
        norm_x = self.attn_norm(x) # pre-norm transformer for its increased training stability
        attn_out = self.attn(norm_x, pad_mask)
        x = x + self.dropout(attn_out)
        # feed-forward component
        norm_x = self.ff_norm(x)
        ff_out = self.down_proj(self.gelu(self.up_proj(norm_x)))
        x = x + self.dropout(ff_out)
        return x


class _MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, dropout_p: float = 0.5):
        super().__init__()
        # Ensure embedding dim can be divided into equally sized heads
        assert(embedding_dim % num_heads == 0)
        # Save parameters and create layers
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.o_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.tensor, pad_mask: torch.tensor = None):
        batch, seq_len, _ = x.shape
        # reshape Q, K, V to (batch, seq_len, num_heads, head_dim) by splitting third dimension (embedding_dim) into separate heads,
        # then transpose to (batch, num_heads, seq_len, head_dim) so that batched matmul (last 2 dims) multiplies head content correctly
        multihead_shape = (batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(multihead_shape).transpose(1, 2)
        k = self.k_proj(x).view(multihead_shape).transpose(1, 2)
        v = self.v_proj(x).view(multihead_shape).transpose(1, 2)

        # 1. multiply Q * K_{transposed} and scale by root of the head dimension
        attn = torch.div(torch.matmul(q, k.transpose(2, 3)), self.head_dim**0.5)
        # apply pad_mask if specified
        if pad_mask is not None:
            attn.masked_fill_(pad_mask, float("-inf"))
        # 2. softmax the scaled, pad-masked attention weights
        attn = torch.softmax(attn, -1)
        # 3. matmul with V to get attention resulting values (and apply dropout for training)
        values = torch.matmul(self.dropout(attn), v)
        # 4. project output values
        out = self.o_proj(values)
        # 5. concatenate separated heads
        # first transpose back to (batch, seq_len, num_heads, head_dim), then change shape to concat num_heads and head_dim
        out = out.transpose(1, 2).view(batch, seq_len, -1)

        return out

