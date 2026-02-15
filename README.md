# Transformer From Scratch (with PyTorch)

This repo contains a **full implementation** of the famous **Transformer architecture** built from scratch using **PyTorch**, from the reasearch paper **"Attention Is All You Need"**, without relying on high level libraries such as HugginFace. This repo not only creates the transformer but also shows all its different usages, such as: **Classification, Text Generation, and Translation.** 

> **This repo is mainly for study and research purposes. Feedback is welcomed :)**  

# Repo Structure
```
Transformer_from_Scratch/
│
├── data/                    # Data used for each model
│
├── models/                  # Saved models (each model have its corresponding Tokenizer and Best Model)  
│
├── notebooks/               # Each notebook contains training phase + trying the saved model
│   ├── classification_train.ipynb       
│   ├── generator_train.ipynb           
│   └── translation_train.ipynb         
│
├── transformer/             # Transformer components
│   ├── decoder.py
│   ├── encoder.py
│   ├── feed_foward.py
│   ├── multi_head_attention.py
│   └── transformer.py       # Full Transformer / Encoder-Decoder model
│
├── sentiment_classifier.py  # Classification / Encoder model
├── text_generator.py        # Text Generation / Decoder model
├── tokenizer.py             # Tokenizer model
├── requirements.txt
├── LICENSE
└── README.md
```
# Index
1. Tokenizer
2. Transformer Architecture
3. Sentiment Classification (Encoder-only)
4. Text Generation (Decoder-only)
5. Translation (Encoder-Decoder)
#### Clone Repo
```bash
git clone https://github.com/JimWid/Transformer-From-Scratch.git
cd Transformer-From-Scratch
```
#### Set Up Virtual Environment
```bash
python -m venv venv
On Windows: venv\Scripts\activate
On Mac: source env/bin/activate
```
#### Install Dependencies
```bash
pip install -r requirements.txt
```
# Tokenizer
I am using a custom tokenizer from Scratch too, but you can use any.
<img width="735" height="366" alt="image" src="https://github.com/user-attachments/assets/240cd9cb-42bc-4060-82c8-32243b6545b2" />

A tokenizer is used to **build a vocabulary** from the dataset, **it maps the words to indices** (it sets a word to a number, **e.g. "the" -> 5**) and it adds special tokens such as:
- ```<PAD>``` # Padding for short sentences to match ```max_len```.
- ```<UNK>``` # Used for Unknown words (words that are not inside vocabulary)
- ```<SOS>``` # Used to declare when a setences begins
- ```<EOS>``` # Used to declare when a sentence ends
> We only use ```<SOS>``` and ```<EOS>``` in Encoder-Decoder models, such as translation.

> NOTE: Tokenizers turns sentences -> tokens, by breaking this down into pieces, my tokenizer split them into whole words, I belive this is easier to understand and visualize.
# Transformer Architecture
Let's first define all the paramerets we are gonna use:
- ```vocab_size```     # Size of vocabulary
- ```embedding_size``` # Vector size of each token
- ```num_layers```     # number of layers
- ```num_heads```      # number of attention-heads
- ```d_ff```           # hidden layer of feed foward network
- ```max_len```        # max length of sentences
- ```dropout```        # dropout

There is 4 main components in a Transformer: **Postional Encoding, Self-Attention, FeedFoward, Masking, LayerNorm**. Let's go over them one by one.
## Positional Encoding
The simpliest way to describe this is adding position information to the embeddings vector, therefore during training, the model doesn't just learn meanings of words, but also their positions. Smart rigth?

We inject position information with the Sinusoidal method:

<img width="267" height="63" alt="image" src="https://github.com/user-attachments/assets/fb1f14c7-2e05-4796-9e31-ca911cb6b2ad" />

#### Postional Encoding Code:
```python
    def get_positional_encoding(self, max_len, embedding_size):
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(embedding_size, dtype=torch.float32).unsqueeze(0)

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2) / embedding_size))
        angle_rads = pos * angle_rates

        positional_encoding = torch.zeros(max_len, embedding_size)

        positional_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2]) # Sin for even values/dimensions
        positional_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2]) # Cos for odd values/dimensions
        
        return positional_encoding.unsqueeze(0)
```

## Self-Attention
Self-Attention is basically a formula:

<img width="364" height="59" alt="image" src="https://github.com/user-attachments/assets/98f9468b-14c7-4f80-a840-5876a07519c0" />

Where Q, K and V are:
- Query (Q)
- Key (K)
- Value (V)

> Each one of these are a matrix full of parameters (weights) that are trainable, with a size of ```embedding_size```.

Inside this equation is it perform what is called **Scaled Dot Product**:

<img width="207" height="268" alt="image" src="https://github.com/user-attachments/assets/88c028d6-f6c9-4473-ba56-eaeebbdb67da" />

#### Multi-Head Attention
Basically concatenating all the outputs of all the attention-heads.

<img width="613" height="240" alt="image" src="https://github.com/user-attachments/assets/c802d6f4-b1fe-4726-8200-430049ea9f91" />

#### Multi-Head Attention Code:

```python
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
            scores = scores.masked_fill(mask == 0, float("-inf"))

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
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, -1, self.embedding_size)

        # Final Linear Projection
        output = self.output(attn_output)

        return output
```

## Feed Foward Network
Feed Foward Network is provides a non-linear and position-wise transformation, it increases model capacity and it mixes the information across features.

<img width="265" height="34" alt="image" src="https://github.com/user-attachments/assets/a5481947-401f-43d8-b7e3-8452a9a42956" />

#### Feed Foward Network Code:
```python
class FeedForward(nn.Module):
    def __init__(self, embedding_size, d_ff, dropout):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(embedding_size, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
```
## Add & LayerNorm
I do not use a custom layer norm, I instead use the PyTorch one, **LayerNorm** is basically another equation, that computes mean and the variance of the features of each token, then it normalizes, turning mean close to 0 and variance close to 1. It has its own parameters **scale and shift**. During training, it changes scale and shift to determine how much to normalize.

<img width="312" height="57" alt="image" src="https://github.com/user-attachments/assets/69f52c76-9838-4672-8107-41bf5bb95c3d" />


We also have **Residual Connections (Add)**, an easy way to see it is that the Residual Connection simply **adds the input to the new changed input**. We are basically adding a correction to the token representation, not a full re-write.

I use LayerNorm and Residual in both Encoder and Decoder:
> Decoder and Encoder are basically wrappers of all these components.

#### Encoder Code:
```python
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
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, d_ff, max_len, dropout, device):
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
```
> Notice how we have an Encoder/Decoder **Block** and then an Encoder/Decoder, this way we can actually determine how many times we want to loop throughout our Encoder/Decoder. 
#### Decoder Code:
```python
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
```
> ```enc_out``` is used for Translation model, and it means the encoded output from the Encoder. I skip it if there is no ```enc_out``` passed to the model. Therefore no **Cross Attention**.

## Full Transformer Code:
```python
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
```
# Sentiment Classification
Once we have our Encoder, we can assign it to do Classification. We do not need the Decoder, since we are not doing anything sequence-to-sequence operation, simply attention.

# Text Generation

# Translation

# License





