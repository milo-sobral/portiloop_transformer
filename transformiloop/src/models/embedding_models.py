from abc import abstractmethod
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, 
    d_model: int, 
    dropout: float = 0.1, 
    max_len: int = 5000):
        """Positional Encoding Module. Adds positional encoding using cosine and sine waves to the input data

        Args:
            d_model (int): Desired size of the Encoding dimension
            dropout (float, optional): dropout value. Defaults to 0.1.
            max_len (int, optional): maximum length of sequence. Defaults to 5000.
        """
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
        
        Returns:
            tensor: returns a tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SineEncoding(nn.Module):
    def __init__(self, 
    in_features: int,
    out_features: int):
        """Method to perform Sine encoding

        Args:
            in_features (int): Number of features of input vector
            out_features (int): Number of desired output features
        """
        super(SineEncoding, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return SineEncoding.t2v(tau, self.f, self.w, self.b, self.w0, self.b0)

    @abstractmethod
    def t2v(tau, f, w, b, w0, b0, arg=None):
        if arg:
            v1 = f(torch.matmul(tau, w) + b, arg)
        else:
            v1 = f(torch.matmul(tau, w) + b)
        v2 = torch.matmul(tau, w0) + b0
        return torch.cat([v1, v2], 1)
