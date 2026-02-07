# Feed Forward Network in the Transformer architecture
"""
    This layer is a simple two-layer fully connected network with a ReLU activation in between.
    It is applied to each position separately and identically, meaning that it does not mix information across different positions in the sequence.
    The feed forward network allows the model to learn complex transformations of the input embeddings,
    enabling it to capture non-linear relationships and interactions between features.

    Architecture:
    FFN(x) = Linear -> ReLU -> Linear -> Dropout

    Mathematically, it can be expressed as:
    FFN(x) = max(0, xW1 + b1)W2 + b2

    Purpose:
    Introduces non-linearity into the transformer.
    Expands the embedding intoa higher-dimensional space, allowing the model to learn more complex representations.
    Projects back to the original embedding size so residual connections remain valid.

    Arguments:
    embedding_size: The dimensionality of the input and output embeddings.
    d_ff: The dimensionality of the inner layer, usually 2-4x larger than embedding_size (the "feed forward" dimension).
    dropout: The dropout rate to apply after the ReLU activation to prevent overfitting.

    Shapes:
    Input: [batch_size, sequence_length, embedding_size]
    Output: [batch_size, sequence_length, embedding_size]
"""

import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embedding_size, d_ff, dropout):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(embedding_size, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))