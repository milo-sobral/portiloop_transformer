from transformiloop.src.models.classifiers.classification_encoder_model import ClassificationModel
from transformiloop.src.utils.train import run
from transformiloop.src.utils.configs import initialize_config, validate_config
import torch
from torchinfo import summary

from transformiloop.src.utils.configs import initialize_config, validate_config
import unittest

def single_experiment():
    config = initialize_config('test')
    if not validate_config(config):
        raise AttributeError("Issue with config.")
    save_model = False
    unique_name = True
    pretrain = False
    finetune_encoder = True

    run(config, 'experiment_enc_newcnn', 'Milo-DEBUG', save_model, unique_name, pretrain, finetune_encoder)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST_CONFIG")

    def test_initialize_model(self):
        classifier = ClassificationModel(self.config)
        summary(
            classifier,
            input_size=[
                (self.config['batch_size'], self.config['seq_len'], self.config['window_size']),
                (self.config['batch_size'], self.config['seq_len']-1)
            ],
            dtypes=[torch.float, torch.float, torch.float],
            depth=3,
        )

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
