import itertools
from transformiloop.src.models.encoding_models import EncodingTypes
from transformiloop.src.models.transformers import ClassificationTransformer, TransformiloopPretrain
# from transformiloop.src.utils.train import run
from transformiloop.src.utils.configs import initialize_config, validate_config
import torch
from torchinfo import summary
from copy import deepcopy
import pprint

from transformiloop.src.utils.configs import initialize_config, validate_config
import unittest


class TestModels(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST_CONFIG")

        combinatorial_config = {
            'duplicate_as_window': [True, False],
            'full_transformer': [True, False],
            'encoding_type': [EncodingTypes.ONE_HOT_ENCODING, EncodingTypes.POSITIONAL_ENCODING],
            'use_cnn_encoder': [True, False],
            'use_last': [True, False]
        }

        self.exclusives = [("duplicate_as_window", "use_cnn_encoder")]

        keys = list(combinatorial_config)
        all_options_iterator = itertools.product(*map(combinatorial_config.get, keys))
        all_options_dicts = [dict(zip(keys, values)) for values in all_options_iterator]
        self.filtered_options = [value for value in all_options_dicts if self.is_ok(value)]

    def update_config(self, config, config_option):
        config = deepcopy(config)
        for key in config_option.keys():
            config[key] = config_option[key]
        return config

    def is_ok(self, config_option):
        for pair in self.exclusives:
            if config_option[pair[0]] and config_option[pair[1]]:
                return False
        return True

    def test_classification_transformer(self):
        for combination in self.filtered_options:
            config = self.update_config(self.config, combination)
            # If we are not using a CNN or duplicate s window, we want to modify d_model for the model to work
            if not config['use_cnn_encoder'] or config['duplicate_as_window']:
                config['d_model'] = (config['window_size'] + config['seq_len']) \
                    if (config['encoding_type'] == EncodingTypes.ONE_HOT_ENCODING) else \
                        config['window_size']
            test_config = validate_config(config)
            self.assertTrue(test_config)
            try:
                classifier = ClassificationTransformer(config)
                in_seq = torch.rand((config['batch_size'], config['seq_len'], config['window_size']))
                in_hist = torch.rand((config['batch_size'], config['seq_len']-1)) if config['full_transformer'] else None
                output = classifier(in_seq, in_hist)
                self.assertEqual(output.shape, torch.Size([self.config['batch_size'], 1]))

            except Exception as e:
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(config)
                print(f"Error during testing of combination {combination}")
                raise e

    def test_pretraining_transformer(self):
        for combination in self.filtered_options:
            config = self.update_config(self.config, combination)
            # If we are not using a CNN or duplicate s window, we want to modify d_model for the model to work
            if not config['use_cnn_encoder'] or config['duplicate_as_window']:
                config['d_model'] = (config['window_size'] + config['seq_len']) \
                    if (config['encoding_type'] == EncodingTypes.ONE_HOT_ENCODING) else \
                        config['window_size']
            test_config = validate_config(config)
            self.assertTrue(test_config)
            try:
                pretrainer = TransformiloopPretrain(config)
                in_seq = torch.rand((config['batch_size'], config['seq_len'], config['window_size']))
                in_hist = torch.rand((config['batch_size'], config['seq_len']-1)) if config['full_transformer'] else None
                in_mask = torch.rand((config['batch_size'], config['seq_len']))
                _, _, out_3 = pretrainer(in_seq, in_hist, in_mask)
                out_1, out_2, _ = pretrainer(in_seq, in_hist, None)

                self.assertEqual(out_1.shape, torch.Size([self.config['batch_size'], 1]))
                self.assertEqual(out_2.shape, torch.Size([self.config['batch_size'], 1]))
                # If we are testing full transformer, then the masked language modeling does not make sense
                if not config['full_transformer']:
                    self.assertEqual(out_3.shape, torch.Size([self.config['batch_size'], self.config['seq_len'], self.config['reconstruction_dim']]))
                
            except Exception as e:
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(config)
                print(f"Error during testing of combination {combination}")
                raise e

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
