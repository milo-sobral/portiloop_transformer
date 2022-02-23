import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


DEFAULT_CONFIG = {
    'N': 6,
    'd_model': 512,
    'd_ff': 2048,
    'h': 8,
    'dropout': 0.1
}


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_encode, tgt_encode, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_encode = src_encode
        self.tgt_encode = tgt_encode
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_encode(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_encode(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.proj(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module.
    This is used to normalize all the weights of individual layers.
    for details, see https://arxiv.org/abs/1607.06450
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # Creates two clones of a Sublayer connection
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # First sublayer connection is used to apply the self attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Second sublayer is used to apply the feedforward
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        # Apply self attention on the output with mask tgt_mask
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Apply attention with query->x and key and values from memory with mask src_mask
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # Apply feed forward
        return self.sublayer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    # Get size of query
    d_k = query.size(-1)
    # Multiply query with key to get scores. Scale scores by squarred size 
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # Nullify the scores for masked positions
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Apply softmax
    p_attn = F.softmax(scores, dim=-1)
    # Apply dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # Multiply by values to get the final attention output.
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Represents a module with multiple attention heads side by side.
    """
    def __init__(self, h, d_model, dropout=0.1):
        '''
        params: 
            h -> number of heads
            d_model -> width of hidden layer
        '''
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        '''
        Adds a hidden layer with dimension d_ff
        '''
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        # Initialize empty vector with dimensions
        pe = torch.zeros(max_len, d_model)
        # Get linspace
        position = torch.arange(0, max_len).unsqueeze(1)
        # Get exponential term in log space
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # apply sine on even positions of array
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine on odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add the positional 'bias' to the input and dropout
        var_tensor = Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        x = x.unsqueeze(-1)
        x = x + var_tensor
        return self.dropout(x)


def make_model(config=DEFAULT_CONFIG):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(config['h'], config['d_model'])
    ff = PositionwiseFeedForward(config['d_model'], config['d_ff'], config['dropout'])
    position = PositionalEncoding(config['d_model'], config['dropout'])
    model = EncoderDecoder(
        Encoder(EncoderLayer(config['d_model'], c(attn), c(ff), config['dropout']), config['N']),
        Decoder(DecoderLayer(config['d_model'], c(attn), c(attn), 
                             c(ff), config['dropout']), config['N']),
        c(position),
        c(position),
        Generator(config['d_model']))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model