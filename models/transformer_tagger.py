# Second Candidate Model - Encoder-only Transformer

import torch
from torch import nn

from model import Model
from vocab import Vocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerTagger(Model):

    def __init__(self, dataset: pd.DataFrame, excluded_fold: int = -1, learning_rate: float = 10**(-3),
                num_blocks: int = 4, embedding_dim: int = 100, num_heads: int = 2, ff_dim: int = 256, dropout_p: float = 0.4):
        if excluded_fold >= 0:
            vocab_set = dataset[dataset["subset"].ne("test") & dataset["fold_id"].ne(excluded_fold)]
        else:
            vocab_set = dataset[dataset["subset"].ne("test")]
        self.vocab = Vocab(vocab_set)
        self.model = Transformer(self.vocab.length(), self.vocab.tag_count(), num_blocks, embedding_dim, num_heads, ff_dim, dropout_p).to(DEVICE)
        self.learning_rate = learning_rate
        print(f"Model Log: Initialised Transformer Model on device {DEVICE}")

    def forward_pass(self, inputs: list) -> list:
        input_ids = torch.tensor([self.vocab.get_token_id(x) for x in inputs], device=DEVICE).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids, torch.tensor([len(inputs)]))
        output_ids = torch.argmax(output, -1).squeeze(0)
        return [self.vocab.get_tag_by_id(x) for x in output_ids]

    # TODO fit and hyperparameter tuning

class Transformer(nn.Module):

    def __init__(self, num_terms: int, num_tags: int, num_blocks: int, embedding_dim: int, num_heads: int, ff_dim: int, dropout_p: float = 0.5):
        super().__init__()
        # word embeddings and positional encoding
        self.embedding = nn.Embedding(num_terms, embedding_dim)
        self.positional = self._get_positional_encoding(embedding_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        # transformer encoder block sequence and outputs
        self.dropout = nn.Dropout(dropout_p)
        self.blocks = nn.ModuleList([_TransformerEncoderBlock(embedding_dim, num_heads, ff_dim, dropout_p) for i in range(num_blocks)])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.out = nn.Linear(embedding_dim, num_tags)

    def forward(self, x: torch.tensor, lengths: torch.tensor):
        # 1. create a padding mask based on the sequence lengths
        batch, seq_len = x.shape
        pad_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.to(x.device).unsqueeze(-1)
        pad_mask = pad_mask.view(batch, 1, 1, seq_len)
        # 2. embed the input IDs
        embedded = self.positional(self.embedding(x))
        # 3. pre-transformer block layer norm and dropout
        embedded = self.dropout(self.embedding_norm(embedded))
        # 4. go through transformer blocks sequentially
        transformer_out = embedded
        for block in self.blocks:
            transformer_out = block(transformer_out, pad_mask)
        # 5. final layernorm and output projection to get tag logits
        out = self.out(self.final_norm(transformer_out))
        return out

    def _get_positional_encoding(self, embedding_dim: int):
        def encoding(x: torch.tensor):
            _, seq_len, _ = x.shape # shape should be (batch, seq_len, embedding_dim)
            # Using sinusoid positional encoding as per Vaswani et al. (2017)
            pos = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(1)
            div = 10000.0 ** (torch.arange(0, embedding_dim, 2, device=x.device, dtype=torch.float32) * 2 / embedding_dim)
            x[..., 0::2] = x[..., 0::2] + torch.sin(pos / div)
            x[..., 1::2] = x[..., 1::2] + torch.cos(pos / div)
            return x
        return encoding

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
        # 4. concatenate separated heads
        # first transpose back to (batch, seq_len, num_heads, head_dim), then change shape to concat num_heads and head_dim
        out = values.transpose(1, 2).reshape(batch, seq_len, -1)
        # 5. project output values
        out = self.o_proj(out)

        return out

