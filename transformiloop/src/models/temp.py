from transformiloop.src.models.transformers import TransformiloopFinetune
from transformiloop.src.utils.configs import initialize_config
from transformiloop.src.models.model_blocks import GRUClassifier
import torch.nn as nn
from torchinfo import summary
import torch


config = initialize_config('Test')
lstm_classifier = GRUClassifier(config).cuda()
transformer_classifier = TransformiloopFinetune(config).cuda()

# print(summary(
#     lstm_classifier, 
#     input_size=[
#         (config['batch_size'], config['seq_len'], config['window_size']),
#         (config['gru_num_layers'], config['batch_size'], config['gru_hidden_size'])]
#         ))

loss = nn.CrossEntropyLoss()

# Get a batch of data
x = torch.rand((config['batch_size'], config['seq_len'], config['window_size'])).cuda()

logits_lstm, _ = lstm_classifier(x, None)
# logits_transformer = transformer_classifier(x, None)
# Get a random target of zeros and ones in a long Tensor
targets = torch.randint(low=0, high=2, size=(config['batch_size'], 1), dtype=torch.float).cuda() 

loss_lstm = loss(logits_lstm, targets)
# loss_transformer = loss(logits_transformer.cpu(), torch.zeros((config['batch_size'], 1)))

loss_lstm.backward()
# loss_transformer.backward()
