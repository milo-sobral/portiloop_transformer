from transformiloop.src.utils.train import run
from transformiloop.src.utils.configs import get_default_config
from transformiloop.src.models.classifiers.classification_encoder_model import ClassificationModel
from torchinfo import summary
import torch

config = get_default_config('test')

save_model = False
unique_name = True
pretrain = False
finetune_encoder = True

print(config)

# summary(
#     model,
#     input_size=[
#         (config['batch_size'], config['seq_len'], config['window_size'])
#     ],
#     dtypes=[torch.float, torch.float, torch.bool],
#     depth=3,
# )

run(config, 'experiment_cnn_stride_fix', 'Milo-DEBUG', save_model, unique_name, pretrain, finetune_encoder)
