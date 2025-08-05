# Tokenizer Architecture
# 1. Fit Function - Done
# 2. Tokenize Text - Done
# 3. Transform - Done
# 4. Pad Sequence - Done

import re
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size=10000, min_freq=1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1} # PAD = Padding is for short sentences
        self.idx2word = {0:"<PAD>", 1: "<UNK>"} # UNK = Unkown, for words that were not in the training phase
    
    def fit(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)

        most_common = word_counts.most_common(self.vocab_size - 2) # Reverse space, taking out 0 and 1
        for idx, (word, count) in enumerate(most_common, start=2): # Starting from 2 
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word 
    
    def tokenize(self, text):
        text = text.lower()
        return re.findall(r"\b\w+\b", text) # Seperates words from sentences, and also removes punctuation

    def transform(self, text):
        if isinstance(text, str):
            tokens = self.tokenize(text)
        else:
            tokens = text
        return [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

    def transform_batch(self, texts):
        return [self.transform(t) for t in texts]

    def pad_sequence(self, sequence, max_len):
        padded = []
        for seq in sequence:
            if len(seq) < max_len:
                # Pad with 0s (the index of <PAD>)
                seq = seq + [self.word2idx["<PAD>"]] * (max_len - len(seq))
            else:
                # Truncate if too long
                seq = seq[:max_len]
            padded.append(seq)
        return padded