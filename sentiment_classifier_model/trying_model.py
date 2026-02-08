import torch
import pickle
from sentiment_classifier import SentimentClassifier

# Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Model parameters
vocab_size = 10_000
embedding_dim = 512
num_layers = 3
num_heads = 4
d_ff = 2048
num_classes = 2  # positive vs negative
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the trained model
checkpoint = torch.load("best_model.pt")
max_len = checkpoint["max_len"]

model = SentimentClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    max_len=max_len,
    num_classes=num_classes,
    dropout=dropout).to(device)

# Load the model state
model.load_state_dict(checkpoint["model_state_dict"])

label_map = {0: "Negative", 1: "Positive"}

def predict_sentiment(sentence, model, tokenizer, max_len, device="cuda"):
    model.eval()

    with torch.no_grad():
        tokens = tokenizer.transform(sentence)
        tokens = tokenizer.pad_sequence([tokens], max_len)
        input_tensor = torch.tensor(tokens, dtype=torch.long).to(device)

        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()
    
        return print(label_map[prediction])

examples = ["I love you!",
            "I can't believe it",
            "I hate this movie so much!", 
            "I think this was wonderful!",
            "Worst experience of my entire life",
            "I don't love you anymore"]

for example in examples:
    predict_sentiment(example, model, tokenizer, max_len)
