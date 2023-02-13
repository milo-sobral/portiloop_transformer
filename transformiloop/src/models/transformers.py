import torch
import torch.nn as nn
from transformiloop.src.models.model_blocks import build_encoder_module, Transformer, get_cnn_embedder


class TransformiloopPretrain(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # CNN based encoder to encode the EEG sequences
        self.cnn_encoder = build_encoder_module(config) if config['use_cnn_encoder'] else None
        # Transformer model which performs window-wise attention
        self.transformer = Transformer(config)
        # Classification head which performs classification 
        self.gender_classifier = nn.Sequential(
            nn.Linear(config['d_model'], config['hidden_mlp']),
            nn.Tanh(),
            nn.Linear(config['hidden_mlp'], 1)
        )
        self.age_regression = nn.Sequential(
            nn.Linear(config['d_model'], config['hidden_mlp']),
            nn.Tanh(),
            nn.Linear(config['hidden_mlp'], 1)
        )

        self.sequence_reconstruction = nn.Sequential(
            nn.Linear(config['d_model'], config['reconstruction_dim']),
            nn.Tanh(),
            nn.Linear(config['hidden_mlp'], config['reconstruction_dim'])
        )

        self.mask_token = nn.Parameter(torch.randn(1, 1, config['embedding_size']))

    def forward(self, x, history, mask):
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

        # Add learnable masking token
        if mask is not None:
            mask = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), x.size(-1))
            x = torch.where(mask != 2, x, self.mask_token)

        # Run through transformer
        x = self.transformer(x, history)

        # Go through classifier
        if self.config['use_last']:
            # We only want to classify based on embedding of last window
            token = x[:, -1]
        else: 
            # We want to classify based on the embedding of CLS token
            token = x[:, 0]

        if mask is None:
            gender = self.gender_classifier(token) # Get gender from token
            age = self.age_regression(token) # Get age from token
            return gender, age, None
        else:
            # Get the reconstructions for windows that have been masked
            if not self.config['full_transformer']:
                x = x[:, 1:]
            seq_reconstructions = self.sequence_reconstruction(x)
            return None, None, seq_reconstructions

    def get_models(self):
        """
        Return the important models.
        This function is meant to be used once pretraining is done to only keep 
        the parts of the pretrained model that we'll reuse later.
        """
        return self.cnn_encoder, self.transformer
        

class TransformiloopFinetune(nn.Module):
    def __init__(self, config, cnn_encoder=None, transformer=None, freeze=False) -> None:
        super().__init__()
        self.config = config
        # CNN based encoder to encode the EEG sequences
        if cnn_encoder is not None:
            self.cnn_encoder = cnn_encoder
        else:
            self.cnn_encoder, seq_len = get_cnn_embedder(config) if config['use_cnn_encoder'] else None
            config['seq_len'] = seq_len

        # Transformer model which performs window-wise attention
        if transformer is not None:
            self.transformer = transformer
        else:            
            self.transformer = Transformer(config)

        # Freeze the pretrained models
        if freeze:
            if self.cnn_encoder is not None:
                for param in self.cnn_encoder.parameters():
                    param.requires_grad = False
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Classification head which performs classification 
        self.classifier = nn.Sequential(
            nn.Linear(config['d_model'], config['hidden_mlp'], device=config['device']),
            nn.ReLU(),
            nn.Linear(config['hidden_mlp'], config['classes'] if config['classes'] > 2 else 1, device=config['device']),
            nn.Sigmoid() if config['classes'] <= 2 else nn.Identity()
        )
    
    def forward(self, x, history):
        batch_dim = x.size(0)
        seq_dim = x.size(1)

        # Encode windows using CNN based model
        if self.cnn_encoder is not None:
            # x = x.contiguous().view(-1, self.config['window_size']).unsqueeze(1)
            x = self.cnn_encoder(x)
            # x = x.view(batch_dim, seq_dim, -1)

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
            token = x[:, -1]
        else: 
            # We want to classify based on the embedding of CLS token
            token = x[:, 0]

        return self.classifier(token)
