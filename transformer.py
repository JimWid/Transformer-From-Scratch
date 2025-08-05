# Project Plan:
# Transformer Architecture

# 1. Multi Head Self-Attention - Done
    # 2. Scaled Dot-Product Attention - Done

# 3. Feed Forward Network - Done
# 4. Encoder Block - Done
# 5. Encoder - Done
    # 6. Positional Encoding - Done

# 7. Decoder Block - Done
# 8. Decoder - Done

# 9. FULL TRANSFORMER - Done

# Hyperparameters:
# N = Number of layers
# embed_size = model dimension
# d_ff = inner-layer dimension
# num_heads = attentions heads
# d_k = attention key dimension
# d_v = attention value dimension
# P_drop = dropout
# L_s = learning Rate

import torch
import math
import torch.nn as  nn
import torch.nn.functional as F
from torch.nn import Embedding

#-----------------------------Multi-Head Attention--------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Linear Projections for Q, K, V
        self.Q = nn.Linear(embed_size, embed_size)
        self.K = nn.Linear(embed_size, embed_size)
        self.V = nn.Linear(embed_size, embed_size)

        # Final output projection
        self.output = nn.Linear(embed_size, embed_size)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q = Query matrix, Shape = [batch, seq_len_q, d_k]
        # K = Key matrix, Shape = [batch, seq_len_k, d_k]
        # V = Value matrix, Shape = [batch, seq_len_k, d_k]

        head_dim = Q.size(-1) # Returns the size of the last dimension of the tensor Q

        # Computing raw attention scores (dot product of Q and K^T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) # Shape: [batch, seq_len_q, seq_len_k]

        # Scaling scores
        scores = scores / math.sqrt(head_dim)

        # Optional mask (used in decoder for casual attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax across key dimesion (last dim (-1) = seq_len_k)
        attn_weights = F.softmax(scores, dim=-1)

        # Multiplying attention weights with V
        output = torch.matmul(attn_weights, V)

        return output

    def forward(self, Q, K, V, mask=None):
        N = Q.shape[0]
        value_len, key_len, query_len = V.shape[1], K.shape[1], Q.shape[1]

        # First: Linear Projection
        Q = self.Q(Q)
        K = self.K(K)
        V = self.V(V)

        # Second: Split into heads
        Q = Q.view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2) # Shape: [length, heads, seq_len, head_dim]
        K = K.view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, value_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Third: Using the scaled dot product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, -1, self.embed_size)

        # Fourth: Final Linear Projection
        output = self.output(attn_output) # Shape: [batch, seq_len, d_model]

        return output

#--------------------------------------Feed-Foward------------------------------------
    
class FeedForward(nn.Module):
    def __init__(self, embed_size, d_ff, dropout):
        super(FeedForward, self).__init__()

        # Structure: FFN(x) = max(0, xW1 + b1)W2 + b2
        # Also written as: FFN(x) = Linear(ReLU(Linear(x)))

        self.linear1 = nn.Linear(embed_size, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x)))) # = FFN
    
#------------------------Transformer Encoder Block-------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, d_ff, dropout):
        super(EncoderBlock, self).__init__()

        # Initializing MHA and FFN
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, d_ff, dropout)

        # Norm Layers
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout2(ffn_output))

        return output

#-----------------------------Full Transfomer Encoder-----------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout, device="cuda"):
        super(Encoder, self).__init__()

        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.get_positional_encoding(max_len, embed_dim).to(device)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):

        # x: [batch_size, seq_len] = this is what x represents

        seq_len = x.size(1)

        assert seq_len <= self.pos_encoding.size(1), "Sequence length exceeds max_len"

        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
    
    def get_positional_encoding(self, max_len, embed_dim):
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1) # creating tensor Shape: (seq_len, 1)
        i = torch.arange(embed_dim, dtype=torch.float32).unsqueeze(0) # creating tensor (1, d_model)

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2) / embed_dim)) # Frequency of each dimentions(sin/cos) Shape: (1, d_model)
        angle_rads = pos * angle_rates # This is the different frequencies to each position Shape: (seq_len, d_model)

        # Now each cell has position * frequency

        # Creating "White Canvas"
        pe = torch.zeros(max_len, embed_dim) # Making Map of Zeros

        # Storing our final positional encoding values
        pe[:, 0::2] = torch.sin(angle_rads[:, 0::2]) # Sin for even values/dimensions
        pe[:, 1::2] = torch.cos(angle_rads[:, 1::2]) # Cos for odd values/dimensions
        
        return pe.unsqueeze(0) # final positional encoding
    

#----------------------------------------Transformer Decoder Block-------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout):
        super(DecoderBlock, self).__init__()

        self.masked_attention = MultiHeadAttention(embed_size, num_heads)
        self.cross_attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_dim)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None):

        _x = self.masked_attention(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(_x))

        _x = self.cross_attention(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(_x))

        _x = self.feed_forward(x)
        output = self.norm3(x + self.dropout(_x))
        return output

#----------------------------------Full Transformer Decoder--------------------------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, device, dropout):
        super(Decoder, self).__init__()

        self.device = device

        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, mask=None):
        N, seq_len = x.shape

        assert seq_len <= self.position_embedding.num_embeddings, "Decoder sequence length exceeds max_len"

        position = torch.arange(0, seq_len, device=self.device).unsqueeze(0).expand(N, seq_len)

        x = self.dropout((self.word_embedding(x) +  self.position_embedding(position)))

        for layer in self.layers:
            x = layer(x, enc_out, src_mask=mask, trg_mask=mask)

        output = self.fc_out(x)
        return output
    
#-------------------------------------------TRANSFORMER----------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, src_pad_idx, trg_pad_idx, embed_dim, num_layers, ff_dim, num_heads, dropout, device, max_len):
        super(Transformer, self).__init__()

        self.encoder = Encoder(vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, device, dropout)

        self.decoder = Decoder(vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, device, dropout)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # Shape: [N, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape # N is batch size

        pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        trg_mask = pad_mask & causal_mask # Shape: [N, 1, trg_len, trg_len]
        return trg_mask
    
    def forward(self, src, trg):

        assert src.size(1) <= self.encoder.pos_encoding.size(1), "Source sequence too long"
        assert trg.size(1) <= self.decoder.position_embedding.num_embeddings, "Target sequence too long"

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        output = self.decoder(trg, enc_src, src_mask=src_mask, trg_mask=trg_mask)
        return output


