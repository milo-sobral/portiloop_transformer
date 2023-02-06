import pathlib
import time
from transformiloop.src.models.transformers import TransformiloopFinetune
from transformiloop.src.utils.configs import initialize_config
from transformiloop.src.models.model_blocks import GRUClassifier
import torch.nn as nn
from torchinfo import summary
import torch
from transformiloop.src.utils.train_utils import finetune_test_epoch_lstm
from transformiloop.src.data.spindle_detection import get_dataloaders



config = initialize_config("test")
config['model_type'] = "lstm"
model = GRUClassifier(config)
model = model.to(config['device'])
print(config['device'])

# dataset_path = pathlib.Path(__file__).parents[2].resolve() / 'dataset'
# _, val_dl, _ = get_dataloaders(config, dataset_path)

# start = time.time()
# finetune_test_epoch_lstm(val_dl, config, model, config['device'], None, 0)
# end = time.time()

# print(f"Finished validation epoch in {end - start} seconds")
