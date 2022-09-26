# from transformiloop.src.models.pretrainers.prediction_encoder_model import PredictionModel
from transformiloop.src.models.classifiers.classification_encoder_model import ClassificationModel, build_encoder_module
from transformiloop.src.models.TFC.encoder import TFC
import torch.nn as nn
import copy

def get_encoder_classifier(config):
    classifier = ClassificationModel(config) 
    encoder = build_encoder_module(config)
    return classifier, encoder
    