from transformiloop.models.prediction_encoder_model import PredictionModel

def get_encoder_based_model(config):
    return PredictionModel(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        dim_hidden=config['dim_hidden'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        prediction_len=config['out_seq_len'],
        seq_len=config['seq_len'],
        latent_dim=config['latent_dim'],
        num_channels_deconv=config['num_channels_deconv'],
        num_layers_deconv=config['num_layers_deconv'],
        device=config['device'])
