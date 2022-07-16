import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformiloop.models.embedding_models import PositionalEncoding


class TransformerExtractor(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int,
                 dim_hidden: int,
                 n_layers: int,
                 dropout: float,
                 device):
        """Transformer Encoder based module which layers positional encoding on a sequence and encodes it using attention mechanism.

        Args:
            d_model (int): Size of encoded dimension
            n_heads (int): Number of heads in each attention layer
            dim_hidden (int): Dimension of hidden neurons for feedforward layers
            n_layers (int): Number of Encoder Layers
            dropout (float): dropout value
        """
        super().__init__()
        self.embedder = nn.Linear(1, d_model, device=device)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, dim_hidden, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model

    def forward(self, x):
        '''
        Args:
            src: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, out_seq_len, d_model]
        '''
        # Add positional encoding
        x = repeat(x, 'b s -> b s e', e=1)
        x = self.embedder(x)
        x = rearrange(x, 'b s e -> s b e')
        x = self.pos_encoder(x)
        x = rearrange(x, 's b e -> b s e')

        # Go through transformer encoder
        x = self.transformer_encoder(x)

        return x


class MLPLatent(nn.Module):
    def __init__(self, 
    latent_dim: int, 
    num_hidden_layers:int, 
    d_model:int, 
    seq_len:int,
    device):
        """Module of the Encoder that transforms output from Transformer based encoder into a latent dimension

        Args:
            latent_dim (int): Size of latent dimension
            num_hidden_layers (int): Number of hidden layers to generate latent dimension
            d_model (int): Size of Transformer encoder dimension
            seq_len (int): Length of input sequence
            device: PyTorch device
        """
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
        """_summary_

        Args:
            x (tensor): Input tensor of dimension [batch_size, seq_len, d_model]

        Returns:
            tensor: Output latent tensor of dimension [batch_size, latent_dim]
        """
        x = self.flatten(x)
        x = self.latent_layers(x)
        return x


class SeqRecCNN(nn.Module):
    def __init__(self, 
    latent_dim: int, 
    num_layers: int, 
    num_channels: int, 
    out_seq_len: int):
        """Recreation Module using Deconvolutions to go from latent dimension to a desired sequence length

        Args:
            latent_dim (int): Size od latent dimension
            num_layers (int): Number of deconvolution layers
            num_channels (int): Number of deconvolution channels
            out_seq_len (int): Desired sequence length
        """
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
        """_summary_

        Args:
            x (tensor): Input tensor of Dimension [batch_size, latent_dim]

        Returns:
            tensor: Output tensor of Dimension [batch_size, out_seq_len] 
        """
        x = repeat(x, 'b l -> b c l', c=1)
        x = self.generator_layers(x)
        return x.squeeze(1)
