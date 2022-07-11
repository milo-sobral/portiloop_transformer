from transformiloop.models.prediction_encoder_model import PredictionModel
from transformiloop.models.classification_encoder_model import ClassificationModel
import torch.nn as nn
import copy


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

def get_encoder_based_classifier(config):
    return ClassificationModel(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        dim_hidden=config['dim_hidden'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        num_classes=config['num_classes'],
        seq_len=config['seq_len'],
        latent_dim=config['latent_dim'],
        device=config['device']
    )

def retrieve_classifier_from_encoder(encoder, config):
    new_encoder = copy.deepcopy(encoder.transformer_extractor)
    new_latent = copy.deepcopy(encoder.latent)
    if config['fix_weights']:
        for param in new_encoder.parameters():
            param.requires_grad = False
        for param in new_latent.parameters():
            param.requires_grad = False
        new_encoder.requires_grad = False
        new_latent.requires_grad = False
    classifier = nn.Linear(config['latent_dim'], config['num_classes'])
    return nn.Sequential(new_encoder, new_latent, classifier)