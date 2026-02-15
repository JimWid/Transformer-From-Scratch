# Full Transformer Architecture
"""
    This module combines the encoder and decoder into a full Transformer model. It includes methods for creating source and target masks,
    and defines the forward pass through the model. The Transformer architecture is designed for sequence-to-sequence tasks,
    such as translation, where the encoder processes the input sequence and the decoder generates the output sequence based on the encoder's output and the target input.

    High Level Overview:
    Source Sentence (src) -> Encoder ->
                            Encoded Representation (enc_src) ->
                            Target Sentence (trg) + Encoded Representation (enc_src) ->
                            Decoder ->
                            Output Sequence

    Architecture:
        src (tokens ids) ->
        Source Mask -> 
        Encoder ->
        Encoded Representation (enc_src) ->
        Target Mask (Padding + Causal) ->
        Decoder ->
        Linear Projection ->
        Vocabulary Logits (output)

    Arguments:
        vocab_size: The size of the vocabulary (number of unique tokens).
        src_pad_idx: The index used for padding in the source sequences.
        trg_pad_idx: The index used for padding in the target sequences.
        embedding_size: The dimensionality of the input and output embeddings.
        num_layers: The number of layers in both the encoder and decoder.
        d_ff: The dimensionality of the feedforward network within each layer.
        num_heads: The number of attention heads to use. Must divide embedding_size evenly.
        dropout: The dropout rate to apply in the model.
        max_len: The maximum length of input and output sequences.
        device: The device (CPU or GPU) on which to run the model.

    Shapes:
        src: [batch_size, src_seq_len]
        trg: [batch_size, trg_seq_len]
        src_mask: [batch_size, 1, 1, src_seq_len]
        trg_mask: [batch_size, 1, trg_seq_len, trg_seq_len]
        enc_src: [batch_size, src_seq_len, embedding_size]
        output: [batch_size, trg_seq_len, vocab_size]
"""

import torch
import torch.nn as nn
from transformer.encoder import Encoder
from transformer.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, src_pad_idx, trg_pad_idx, embedding_dim, num_layers, d_ff, num_heads, max_len, dropout, device):
        super(Transformer, self).__init__()

        self.encoder = Encoder(vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len,dropout, device)
        self.decoder = Decoder(vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)

        enc_out = self.encoder(src, src_mask)

        output = self.decoder(trg, enc_out, src_mask)
        return output
