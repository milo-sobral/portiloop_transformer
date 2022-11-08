import itertools
from transformiloop.src.models.encoding_models import EncodingTypes
from transformiloop.src.models.transformers import ClassificationTransformer
from transformiloop.src.utils.train import run
from transformiloop.src.utils.configs import initialize_config, validate_config
import torch
from torchinfo import summary
from copy import deepcopy
import pprint

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


class TestModels(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST_CONFIG")

    def test_cnn_class_model(self):
        combinatorial_config = {
            'duplicate_as_window': [True, False],
            'full_transformer': [True, False],
            'encoding_type': [EncodingTypes.ONE_HOT_ENCODING, EncodingTypes.POSITIONAL_ENCODING],
            'use_cnn_encoder': [True, False],
            'use_last': [True, False]
        }

        exclusives = [("duplicate_as_window", "use_cnn_encoder")]

        def update_config(config, config_option):
            config = deepcopy(config)
            for key in config_option.keys():
                config[key] = config_option[key]
            return config

        def is_ok(config_option):
            for pair in exclusives:
                if config_option[pair[0]] and config_option[pair[1]]:
                    return False
            return True

        keys = list(combinatorial_config)
        all_options_iterator = itertools.product(*map(combinatorial_config.get, keys))
        all_options_dicts = [dict(zip(keys, values)) for values in all_options_iterator]
        filtered_options = [value for value in all_options_dicts if is_ok(value)]
        
        for combination in filtered_options:
            config = update_config(self.config, combination)
            # If we are not using a CNN or duplicate s window, we want to modify d_model for the model to work
            if not config['use_cnn_encoder'] or config['duplicate_as_window']:
                config['d_model'] = (config['window_size'] + config['seq_len']) \
                    if (config['encoding_type'] == EncodingTypes.ONE_HOT_ENCODING) else \
                        config['window_size']
            test_config = validate_config(config)
            self.assertTrue(test_config)
            try:
                classifier = ClassificationTransformer(config)
                summary(
                    classifier,
                    input_size=[
                        (config['batch_size'], config['seq_len'], config['window_size']),
                        (config['batch_size'], config['seq_len']-1)
                    ],
                    dtypes=[torch.float, torch.float, torch.float],
                    depth=3,
                    verbose=0
                )
            except Exception as e:
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(config)
                print(f"Error during testing of combination {combination}")
                raise e

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
