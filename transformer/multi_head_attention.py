# Multi-Head Attention implementation
"""
    This layer allows each token to attend to every other token in the sequence
    using multiple attention heads, enabling the model to capture different
    relational patterns simultaneously.

    Architercture:
    Input -> Linear Projections (Q, K, V)
          -> Split into Multiple Heads
          -> Scaled Dot-Product Attention for each head
          -> Concatenate Heads
          -> Final Linear Projection

    Arguments:
    embedding_size: The dimensionality of the input and output embeddings.
    num_heads: The number of attention heads to use. Must divide embedding_size evenly.

    Shapes:
    Input: [batch_size, sequence_length, embedding_size]
    Output: [batch_size, sequence_length, embedding_size]
"""

import torch
import math
import torch.nn as  nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        # Linear Projections for Q, K, V
        self.Q = nn.Linear(embedding_size, embedding_size)
        self.K = nn.Linear(embedding_size, embedding_size)
        self.V = nn.Linear(embedding_size, embedding_size)

        # Final output projection
        self.output = nn.Linear(embedding_size, embedding_size)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q = Query matrix, Shape = [batch, seq_len_q, d_k]
        # K = Key matrix, Shape = [batch, seq_len_k, d_k]
        # V = Value matrix, Shape = [batch, seq_len_k, d_k]

        head_dim = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) 

        # Scaling scores
        scores = scores / math.sqrt(head_dim)

        # Optional mask
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Multiplying attention weights with V
        output = torch.matmul(attn_weights, V)

        return output

    def forward(self, Q, K, V, mask=None):

        # Getting Batch Size and Sequence Lengths
        N = Q.shape[0]
        value_len, key_len, query_len = V.shape[1], K.shape[1], Q.shape[1]

        # Linear Projection
        Q = self.Q(Q)
        K = self.K(K)
        V = self.V(V)

        # Spliting into heads
        Q = Q.view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, value_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Using the scaled dot product attention
        if mask is not None:
            mask = mask.expand(N, self.num_heads, mask.size(-2), mask.size(-1))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, -1, self.embedding_size)

        # Final Linear Projection
        output = self.output(attn_output)

        return output