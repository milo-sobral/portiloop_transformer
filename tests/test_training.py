from transformiloop.src.utils.train import run
from transformiloop.src.utils.configs import get_default_config
import torch


def single_experiment():
    config = get_default_config('test')

    save_model = False
    unique_name = True
    pretrain = False
    finetune_encoder = True

    run(config, 'experiment_full_posenc', 'Milo-DEBUG', save_model, unique_name, pretrain, finetune_encoder)


def test_training():
    # Get the config
    name = 'test'
    config = get_default_config(name)

    # Set some variables for testing
    save_model = False
    unique_name = True
    pretrain = False
    finetune_encoder = True
    initial_validation = False

    # Modify config so testing does not last too long
    config['epochs'] = 1
    config['log_every'] = 1
    config['batches_per_epoch'] = 10
    config['max_val_batches'] = config['seq_len'] + 3

    # Run training for one short epoch to check if everything goes well
    run(config, 'TESTING_TRANSFORMILOOP', 'Milo-TESTING', save_model, unique_name, pretrain, finetune_encoder, initial_validation)

if __name__ == '__main__':
    single_experiment()
