import wandb
import torch
import torch.optim as optim
import os

from transformiloop.src.models.transformers import TransformiloopPretrain
from transformiloop.src.utils.train_utils import WarmupTransformerLR

def test_wandb(pretrained_model):
    os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"  # TODO insert my own key
    api = wandb.Api()
    run = api.run(pretrained_model['run_path'])
    config = run.config
    model_dict = torch.load(\
            wandb.restore(pretrained_model['model_name'], run_path=pretrained_model['run_path']).name)
    # Initialize the model
    model = TransformiloopPretrain(config)
    model_state_dict = model_dict['model']
    model_state_dict = {k: v for k, v in model_state_dict.items() if "transformer.positional_encoder.pos_encoder.pe" != k}
    # Get the latest model weights filename, the highest multiple of save_every which is less than the last batch
    model.load_state_dict(model_state_dict)
    model.to(config['device'])
    
    cnn_encoder, transformer = model.get_models()

if __name__ == "__main__":
    pretrained_model = {
        'run_path': 'portiloop/portiloop/EXPERIMENT_1',
        'model_name': 'model_1240000.ckpt'
    }
    test_wandb(pretrained_model)