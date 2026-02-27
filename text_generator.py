from transformer.decoder import Decoder
import torch.nn as nn

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, d_ff, device, max_len, dropout=0.1):
        super().__init__()
        
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            device=device,
            max_len=max_len,
            dropout=dropout
        )

        self.generator = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        decoded = self.decoder(x)
        logits = self.generator(decoded)
        return logits