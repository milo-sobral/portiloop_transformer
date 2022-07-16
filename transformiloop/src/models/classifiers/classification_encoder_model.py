from torch import tensor
import torch.nn as nn
from transformiloop.src.models.helper_models import TransformerExtractor, MLPLatent, SeqRecCNN

class ClassificationModel(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dim_hidden: int,
        n_layers: int, 
        seq_len: int, 
        device,
        dropout: float = 0.5):
        """Prediction module to build models used for pretraining on EEG data.

        Args:
            d_model (int): Dimension of the hidden layers
            n_heads (int): Number of heads for each encoder layer
            dim_hidden (int): Dimension of the Feedforward layer for each TRansformer Encoder Layer
            n_layers (int): Number of Transformer encoder layers
            prediction_len (int): Length of the unseen predicted sequence
            seq_len (int): Length of the seen recreated sequence for Autoencoder  
            latent_dim (int): Dimension of the hidden latent dimension fro autoencoder
            num_channels_deconv (int): Number of Channels in the deconvolution layers
            num_layers_deconv (int): Number of layers in the deconvolution
            dropout (float, optional): Defaults to 0.5.
        """
        super().__init__()

        self.transformer_extractor = TransformerExtractor(d_model=d_model,
                                                          n_heads=n_heads,
                                                          dim_hidden=dim_hidden,
                                                          n_layers=n_layers,
                                                          dropout=dropout,
                                                          device=device)       

        # self.latent = MLPLatent(num_classes, 1, d_model, seq_len, device)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(d_model*seq_len, 1)

    def forward(self, x: tensor):
        """_summary_

        Args:
            x (tensor): Input Tensor of dimension [batch_size, seq_len]

        Returns:
            x_pred (tensor): Output tensor of Dimension [batch_size, prediction_len] after going through the Autoencoder model
            x_rec (tensor): Output tensor of Dimension [batch_size, seq_len] after going through the Autoencoder model
        """

        # Go through feature extractor
        x = self.transformer_extractor(x)

        # Get latent vector
        # x = self.latent(x)
        x = self.flatten(x)
        # # Generate output signal from latent vector
        x = self.classifier(x)
        return x
