# Transformer + Sentiment Classifier From Scratch

A full implementation and creation of a Transformer architecture built from scratch, from the reasearch paper "Attention Is All You Need".
Used to train a Sentiment Classifier on Amazon/IMDb/Yelp reviews dataset.

# Highlights
Fully custom Transformer Encoder and Decoder implementation in PyTorch.
- Trained a sentiment classifier (positive/negative) using only the Transformer **encoder**.
- Custom tokenizer built from scratch with padding, batching, and vocabulary creation.
- Implements early stopping and model checkpointing for optimal performance.
- Final model achieves **~90% training accuracy** and **~80% validation accuracy**.
- Manually tested inference correctly classifies 5 out of 6 unseen inputs most tries.

# Usage
```bash
git clone https://github.com/JimWid/Transformer-From-Scratch.git
cd Transformer-From-Scratch
```
#### Set Up Virtual Environment
```bash
python -m venv env
On Windows: .\env\Scripts\activate
              or 
On Mac: source env/bin/activate
pip install -r requirements.txt
```
#### Start Training and Evaluate (Optional)
```bash
python train.py
python trying_model.py
```

# The Purpose
The whole purpuse of this project was to simply have a deep understanding of what happens under the hood in NLP models such as chatGPT.
It serves me personally as an educational AI project to showcase the ability to build and implement research papers models.

# Explanation / My Understanding
transformer.py
- **MultiHeadAttention**: It implements the attention mechanism that allows the model to wrigh the importance of each word in a sentece relative to others.
  - Projects inputs into Query (Q), Key(K), and Value(V) matrices.
  - Splits Q, K, V into num_heads subspaces for parallel attention (multi-head).
  - Recombines the heads and applies a final linear projection.
  - Function:
  
    <img width="365" height="54" alt="Screenshot 2025-08-04 224157" src="https://github.com/user-attachments/assets/e880be13-167c-43aa-b015-96abaac3301b" />
 
- **FeedForward**: A position-wise feedforward neural network (FFN) applied after attention, is used to transform token representations.
  - Architecture:
      - FFN(x) = Linear(ReLU(Linear(x)))
  - Applies a non-linear transformation independently to each token.
  - Includes dropout for regularization.
    
- **Encoder**: One layer/block of the encoder consisting of:
  - Multi-head self-attention
  - Feed-forward network
  - Residual connections
  - Layer normalization

- **Encoder**: Stack of EncoderBlocks + input embeddings and positional encodings.
  - nn.Embedding: Converts token indices into dense vectors.
  - get_postional_encoding(): Computes sinusoidal(cos/sin) positional encodings to inject word order.
  - ModuleList: Holds the stack of encoder layers.
 
- **DecoderBlock**: One layer/block of the decoder, consisting of:
  - Masked self-attention (for autoregression)
  - Cross-attention (attends to encoder output)
  - Feed-forward network
  - Residual connections & normalization
 
- **Decoder**: Stack of DecoderBlocks plus embeddings and positional encodings.
  - Word embeddings and position embeddings.
  - Decoder blocks.
  - Final output projection (fc_out) to map hidden states back to vocabulary.
 
- **Transformer**: Puts the encoder and decoder together into a full Transformer model + creates masks for padding tokens and causal masks.

sentiment_classifier.py
- Uses only the encoder of the Transformer.
- Performs mean pooling over token embeddings to classify sentence-level sentiment.
- Classification head: Linear -> ReLU -> Dropout -> Linear.

Hyperparametes:

<img width="264" height="179" alt="Screenshot 2025-08-04 225758" src="https://github.com/user-attachments/assets/124b7638-9662-4111-86ae-c638bd782b52" />

# File Structure
```bash
├── train.py                # Training loop with early stopping
├── trying_model.py         # Inference/testing script
├── transformer.py          # Full transformer implementation
├── sentiment_classifier.py # Encoder + classification head
├── tokenizer.py            # Custom tokenizer and padding logic
├── data/                   # (Optional) You can use your own data files if you want :b
├── tokenizer.pkl           # Saved tokenizer with learned vocabulary
├── best_model.pt           # Saved best model
└── README.md               # Project documentation
```

# Results
#### Epoch 20 Final Results:

<img width="613" height="119" alt="Screenshot 2025-08-04 195630" src="https://github.com/user-attachments/assets/8f1bd392-4329-46cf-861a-e61a86c58aa9" />

#### Evaluation with my sentences:

<img width="445" height="277" alt="Screenshot 2025-08-04 195703" src="https://github.com/user-attachments/assets/46fcd243-843b-402d-9583-1d8a225c4c7d" />

#### Metrics:
<img width="1169" height="491" alt="Screenshot 2025-08-04 230220" src="https://github.com/user-attachments/assets/7fa1af3a-9ec3-41c9-8328-0e3b4bff37b5" />

#### Validation Confusion Matrix:
<img width="617" height="475" alt="image" src="https://github.com/user-attachments/assets/740bd791-f046-4d16-adaf-5c6af9000631" />







