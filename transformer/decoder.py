# DecoderBlock and Decoder
"""
    DecoderBlock: This is the basic building block of the decoder.
                  It consists of a masked multi-head attention layer, a cross-attention layer,
                  and a feed-forward neural network. Each of these layers is followed by a layer normalization and dropout for regularization.

    Decoder: The Decoder is a stack of multiple DecoderBlocks. It takes the target sequence,
             applies an embedding layer to convert tokens into dense vectors,
             adds positional encoding to retain the order of the sequence, and then passes the result through the stack of DecoderBlocks.
             Finally, it applies a linear layer to project the output to the vocabulary size.

    Architecture:
    Input -> Masked Multi-Head Attention (prevents tokens from attenting to future tokens)
          -> Add & Norm
          -> Cross-Attention (Encoder-Decoder Attention)
          -> Add & Norm
          -> Feed Forward
          -> Add & Norm
          -> Output

    Key Mechanism:
    Masked Multi-Head Attention: Ensures autoregressive behavior by restricting each position to attend
                                 only to previous tokens. Critical for training models that perform
                                 next-token prediction.
    Cross-Attention: Allows the decoder to attend to the encoder's output,
                     enabling it to incorporate information from the input sequence when generating the output.

    Arguments:
    vocab_size: The size of the target vocabulary.
    embedding_size: The dimensionality of the input and output embeddings.
    num_layers: The number of DecoderBlocks to stack in the Decoder.
    num_heads: The number of attention heads to use in the Multi-Head Attention layers.
    d_ff: The dimensionality of the feed-forward network's inner layer.
    max_len: The maximum length of the target sequences (used for positional encoding).
    device: The device (CPU or GPU) on which the model will run.
    dropout: The dropout rate for regularization.

    Shapes:
    Input: [batch_size, target_sequence_length]
    Output: [batch_size, target_sequence_length, vocab_size]
"""

import torch
import torch.nn as nn
from transformer.multi_head_attention import MultiHeadAttention
from transformer.feed_foward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, d_ff, dropout):
        super(DecoderBlock, self).__init__()

       # Initializing MHA and FFN
        self.masked_attention = MultiHeadAttention(embedding_size, num_heads)
        self.cross_attention = MultiHeadAttention(embedding_size, num_heads)
        self.feed_forward = FeedForward(embedding_size, d_ff, dropout)
        
        # Norm Layers
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out=None, src_mask=None, trg_mask=None):

        # Masked Multi-Head Attention
        _x = self.masked_attention(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(_x))

        # Cross-Attention
        if enc_out != None:
            _x = self.cross_attention(x, enc_out, enc_out, src_mask)
            x = self.norm2(x + self.dropout(_x))

        # Feed Forward
        _x = self.feed_forward(x)
        output = self.norm3(x + self.dropout(_x))
        return output

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, d_ff, max_len, dropout, device):
        super(Decoder, self).__init__()

        self.device = device
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embedding_size, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Causal Mask
        self.register_buffer("trg_mask", torch.tril(
            torch.ones(max_len, max_len, device=self.device)
            ).bool())
        
        # Final Linear Projection
        self.fc_out = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, enc_out=None, src_mask=None):

        # x: [batch_size, target_seq_len]
        N, seq_len = x.shape
        mask = self.trg_mask[:seq_len, :seq_len]

        position = torch.arange(0, seq_len, device=self.device).unsqueeze(0).expand(N, seq_len)

        # x: [batch_size, target_seq_len, embedding_size]
        x = self.dropout((self.word_embedding(x) +  self.position_embedding(position)))

        # Pass through each DecoderBlock
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, mask)

        logits = self.fc_out(x)

        return logits