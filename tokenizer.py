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
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3} # PAD = Padding is to fill the gaps in short sentences
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"} # UNK = Unkown, for words that were not in the training phase
    
    def fit(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)

        most_common = word_counts.most_common(self.vocab_size - 4) # Reverse space, taking out 0, 1, 2 and 3
        for idx, (word, count) in enumerate(most_common, start=4): # Starting from 4 
            if count >= self.min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word 
    
    def tokenize(self, text):
        text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text) # Seperates words from sentences, and also removes punctuation

    def transform(self, text): # Use this for single text
        if isinstance(text, str):
            tokens = self.tokenize(text)
        else:
            tokens = text
        return [self.word_to_idx.get(t, self.word_to_idx["<UNK>"]) for t in tokens]

    def pad_sequence(self, sequence, max_len):
            if len(sequence) < max_len:
                # Pad with 0s (the index of <PAD>)
                sequence = sequence + [self.word_to_idx["<PAD>"]] * (max_len - len(sequence))
            else:
                # Truncate if too long
                sequence = sequence[:max_len]
            return sequence