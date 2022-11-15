import torch
import torch.nn as nn
from transformiloop.src.models.model_blocks import build_encoder_module, Transformer


class ClassificationTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # CNN based encoder to encode the EEG sequences
        self.cnn_encoder = build_encoder_module(config) if config['use_cnn_encoder'] else None
        # Transformer model which performs window-wise attention
        self.transformer = Transformer(config)
        # Classification head which performs classification 
        self.classifier = nn.Sequential(
            nn.Linear(config['d_model'], config['hidden_mlp']),
            nn.Tanh(),
            nn.Linear(config['hidden_mlp'], 1)
        )

    def forward(self, x, history):
        batch_dim = x.size(0)
        seq_dim = x.size(1)

        # Encode windows using CNN based model
        if self.cnn_encoder is not None:
            x = x.contiguous().view(-1, self.config['window_size']).unsqueeze(1)
            x = self.cnn_encoder(x)
            x = x.view(batch_dim, seq_dim, -1)

        # Copies first element of window over the whole window
        if self.config['duplicate_as_window']:
            A = torch.ones((batch_dim, seq_dim, self.config['embedding_size']))
            B = x[:, :, -1]
            x = A * B.unsqueeze(-1)  

        # Run through transformer
        x = self.transformer(x, history)

        # Go through classifier
        if self.config['use_last']:
            # We only want to classify based on embedding of last window
            x = x[:, -1]
        else: 
            # We want to classify based on the embedding of CLS token
            x = x[:, 0]
        x = self.classifier(x)
        return x
