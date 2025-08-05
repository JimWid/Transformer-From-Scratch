from transformer import Encoder
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, num_classes, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_len=max_len,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, src_mask=None):
        # x: [batch_size, seq_len]
        x = self.encoder(x, src_mask) # [batch_size, seq_len, embed_dim]

        # Mean pooling of the sequence_length (dim=1)
        x = x.mean(dim=1)
        #x = nn.AdaptiveAvgPool1d(1)

        logits = self.classifier(x)
        return logits