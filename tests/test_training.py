from transformiloop.src.utils.configs import initialize_config, validate_config
from transformiloop.src.models.transformers import ClassificationTransformer, TransformiloopPretrain
from transformiloop.src.models.encoding_models import EncodingTypes
from transformiloop.src.utils.train_utils import run_pretrain_batch, seq_rec_loss
import torch
import torch.nn as nn
import unittest

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST_CONFIG")

    def test_pretraining_batch(self):
        self.assertNotEqual(self.config, None)

        mse_loss = nn.MSELoss()
        losses = {
            'gender': nn.BCEWithLogitsLoss(),
            'age': mse_loss,
            'seq_rec': (lambda pred, exp, mask: seq_rec_loss(pred, exp, mask, mse_loss))
        }

        # Initialize the model with all the right config to Pretrain
        self.config['duplicate_as_window'] = False
        self.config['full_transformer'] = False
        self.config['encoding_type'] = EncodingTypes.POSITIONAL_ENCODING,
        self.config['use_cnn_encoder'] = True
        self.config['use_last'] = False
        model = TransformiloopPretrain(self.config)
        model.train()

        # Generate a fake batch
        batch = (
            torch.rand((self.config['batch_size'], self.config['seq_len'], self.config['window_size'])),
            torch.randint(low=0, high=1, size=(self.config['batch_size'], 1)),
            torch.rand((self.config['batch_size'], 1)),
            torch.randint(low=0, high=3, size=(self.config['batch_size'], self.config['seq_len'])),
            torch.rand((self.config['batch_size'], self.config['seq_len'], self.config['window_size']))
        )
        loss, losses, predictions = run_pretrain_batch(batch, model, losses)
        loss.backward()
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()