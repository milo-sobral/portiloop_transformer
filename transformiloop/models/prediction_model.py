import torch.nn as nn
from transformiloop.models.helper_models import TransformerExtractor, MLPLatent

class PredictionModel(nn.Module):
    def __init__(self, d_model: int, 
                 n_heads: int, 
                 dim_hidden: int,
                 n_layers: int, 
                 prediction_len: int, 
                 seq_len: int, 
                 latent_dim: int, 
                 num_channels_deconv: int, 
                 num_layers_deconv: int,
                 dropout: float = 0.5):
        super().__init__()

        self.transformer_extractor = TransformerExtractor(d_model=d_model,
                                                          n_heads=n_heads,
                                                          dim_hidden=dim_hidden,
                                                          n_layers=n_layers,
                                                          dropout=dropout)       

        # self.latent = MLPLatent(latent_dim, 2, d_model, seq_len)

        # self.predictor = SeqRecCNN(latent_dim, num_layers_deconv, num_channels_deconv, prediction_len)
        # self.recreator = SeqRecCNN(latent_dim, num_layers_deconv, num_channels_deconv, seq_len)

        self.predictor = MLPLatent(prediction_len, 0, d_model, seq_len)
        self.recreator = MLPLatent(seq_len, 0, d_model, seq_len)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [batch_size, out_seq_len]
        """

        # Go through feature extractor
        x = self.transformer_extractor(x)

        # Get latent vector
        # x = self.latent(x)

        # Generate output signal from latent vector
        x_rec = self.recreator(x)
        x_pred = self.predictor(x)
        return x_pred, x_rec