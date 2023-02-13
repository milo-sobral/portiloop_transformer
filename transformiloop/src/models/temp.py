import pathlib
import time
from transformiloop.src.data.spindle_trains import get_dataloaders_spindle_trains
from transformiloop.src.models.transformers import TransformiloopFinetune
from transformiloop.src.utils.configs import initialize_config
from transformiloop.src.models.model_blocks import GRUClassifier
import torch.nn as nn
from torchinfo import summary
import torch
from transformiloop.src.utils.train_utils import finetune_epoch, finetune_test_epoch_lstm
from transformiloop.src.data.spindle_detection import get_dataloaders

import torch.optim as optim


config = initialize_config("test")
config['model_type'] = "transformer"
config['classes'] = 2
# model = GRUClassifier(config)
model = TransformiloopFinetune(config)
model = model.to(config['device'])
print(config['device'])

dataset_path = pathlib.Path(__file__).parents[2].resolve() / 'dataset'
MASSdataset_path = dataset_path / 'MASS_preds'
train_dl, val_dl = get_dataloaders_spindle_trains(MASSdataset_path, dataset_path, config)
optimizer = optim.AdamW(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"],
)
start = time.time()
finetune_epoch(train_dl, config, config['device'], model, optimizer, None, None, 0)
end = time.time()
print(f"Finished training epoch in {end - start} seconds")


start = time.time()
finetune_test_epoch_lstm(val_dl, config, model, config['device'], None, 0)
end = time.time()

print(f"Finished validation epoch in {end - start} seconds")
