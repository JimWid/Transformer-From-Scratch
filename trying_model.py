import torch
import pickle

#----------------------------------
from sentiment_classifier import SentimentClassifier
#----------------------------------

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

#--------------Loading Tokenizer----------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = 10_000
embed_dim = 512
num_layers = 3
num_heads = 4
ff_dim = 2048
num_classes = 2  # positive vs negative
dropout = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint = torch.load("best_model.pt")
max_len = checkpoint["max_len"]

model = SentimentClassifier(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    ff_dim=ff_dim,
    max_len=max_len,
    num_classes=num_classes,
    dropout=dropout).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

label_map = {0: "Negative", 1: "Positive"}

def predict_sentiment(sentence, model, tokenizer, max_len, device="cuda"):
    model.eval()

    with torch.no_grad():
        tokens = tokenizer.transform(sentence)
        #print("Transformed:", tokens)
        tokens = tokenizer.pad_sequence([tokens], max_len)
        #print("Padded:", tokens)

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
