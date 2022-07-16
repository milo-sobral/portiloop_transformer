import torch.nn as nn
import torch

class target_classifier(nn.Module): # Frequency domain encoder
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*configs['final_out_channels'], 64)
        self.logits_simple = nn.Linear(64, 1)

    def forward(self, emb):
        # """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred