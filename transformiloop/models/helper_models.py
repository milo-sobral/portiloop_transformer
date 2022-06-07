import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerExtractor(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads,
                 dim_hidden,
                 n_layers,
                 dropout):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, dim_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model

    def forward(self, x):
        '''
        Args:
            src: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, out_seq_len]
        '''
        # Add positional encoding
        x = repeat(x, 'b s -> s b e', e=1)
        x = self.pos_encoder(x)

        # Go through transformer encoder
        x = self.transformer_encoder(x)
        x = rearrange(x, 's b e -> b s e')

        return x


class MLPLatent(nn.Module):
    def __init__(self, latent_dim, num_hidden_layers, d_model, seq_len):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Hidden layers
        linears = []
        for i in range(num_hidden_layers-1):
            linears.append(nn.Linear(d_model, d_model, device=device))

        # Add to sequential
        self.latent_layers = nn.Sequential(nn.Linear(d_model * seq_len, d_model, device=device))
        self.latent_layers.append(nn.ReLU())

        for layer in linears:
            self.latent_layers.append(layer)
            self.latent_layers.append(nn.ReLU())
        self.latent_layers.append(nn.Linear(d_model, latent_dim, device=device))
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.latent_layers(x)
        return x


class SeqRecCNN(nn.Module):
    def __init__(self, latent_dim, num_layers, num_channels, out_seq_len):
        super().__init__()

        assert num_channels % 2**num_layers == 0, "Number of channels must be a power of 2"
        assert (out_seq_len - latent_dim) % num_layers == 0, "Difference between input and output length must be divisible by number of layers"

        layers = []
        diff_layer = (out_seq_len - latent_dim) // num_layers

        layers.append(nn.ConvTranspose1d(
                in_channels=1,
                out_channels=num_channels, 
                kernel_size=diff_layer+1,
                stride=1, 
                padding=0, 
                bias=False))
        layers.append(nn.BatchNorm1d(num_channels))
        layers.append(nn.ReLU(True))

        for i in range(num_layers)[1:]:
            layers.append(nn.ConvTranspose1d(
                in_channels=num_channels//2**(i-1),
                out_channels=num_channels//2**i, 
                kernel_size=diff_layer+1,
                stride=1, 
                padding=0, 
                bias=False))
            layers.append(nn.BatchNorm1d(num_channels//2**i))
            layers.append(nn.ReLU(True))
        
        layers.append(nn.ConvTranspose1d(
                in_channels=num_channels//2**(num_layers-1), 
                out_channels=1, 
                kernel_size=3,
                padding=1, 
                bias=False))

        self.generator_layers = nn.Sequential()
        for layer in layers:
            self.generator_layers.append(layer)
    
    def forward(self, x):
        x = repeat(x, 'b l -> b c l', c=1)
        x = self.generator_layers(x)
        return x.squeeze(1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)