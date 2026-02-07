# EnconderBlock and Decoder
"""
    EncoderBlock: This is the basic building block of the encoder.
                It consists of a multi-head attention layer followed by a feed-forward neural network.
                Each of these layers is followed by a layer normalization and dropout for regularization.

    Encoder: The Encoder is a stack of multiple EncoderBlocks. It takes the input sequence,
            applies an embedding layer to convert tokens into dense vectors,
            adds positional encoding to retain the order of the sequence, and then passes the result through the stack of EncoderBlocks.

    Architecture:
    Input -> Multi-Head Attention
          -> Add & Norm
          -> Feed Forward
          -> Add & Norm
          -> Output

    Arguments:
    embedding_size: The dimensionality of the input and output embeddings.
    num_layers: The number of EncoderBlocks to stack in the Encoder.
    num_heads: The number of attention heads to use in the Multi-Head Attention layer.
    d_ff: The dimensionality of the feed-forward network's inner layer.
    max_len: The maximum length of the input sequences (used for positional encoding).
    dropout: The dropout rate for regularization.

    Shapes:
    Input: [batch_size, sequence_length]
    Output: [batch_size, sequence_length, embedding_size]
"""

import torch
import torch.nn as nn
from torch.nn import Embedding
from .multi_head_attention import MultiHeadAttention
from .feed_foward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, d_ff, dropout):
        super(EncoderBlock, self).__init__()

        # Initializing MHA and FFN
        self.attention = MultiHeadAttention(embedding_size, num_heads)
        self.feed_forward = FeedForward(embedding_size, d_ff, dropout)

        # Norm Layers
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Attention
        attn_output = self.attention(x, x, x, mask)

        # Add & Norm
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed Forward
        ffn_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout2(ffn_output))

        return output

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, d_ff, max_len, dropout, device="cuda"):
        super(Encoder, self).__init__()

        self.embedding = Embedding(vocab_size, embedding_size)
        self.pos_encoding = self.get_positional_encoding(max_len, embedding_size).to(device)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(embedding_size, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):

        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
    
    def get_positional_encoding(self, max_len, embedding_size):
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(embedding_size, dtype=torch.float32).unsqueeze(0)

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2) / embedding_size))
        angle_rads = pos * angle_rates

        positional_encoding = torch.zeros(max_len, embedding_size)

        positional_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2]) # Sin for even values/dimensions
        positional_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2]) # Cos for odd values/dimensions
        
        return positional_encoding.unsqueeze(0)