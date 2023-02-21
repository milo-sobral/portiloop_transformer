import logging
from copy import deepcopy
from math import floor
from random import choices, uniform, gauss
import torch
from transformiloop.src.models.encoding_models import EncodingTypes

EPSILON_NOISE = 0.25 # Proportion of samples which are fully random


def initialize_config(name):
    global DEFAULT_CONFIG
    config = deepcopy(DEFAULT_CONFIG)
    config['exp_name'] = name
    validate_config(config)
    return config


DEFAULT_CONFIG = {
    # Data params
    'batch_size' : 256,
    'seq_len': 50,
    'window_size': 54,
    'seq_stride': 54,
    'network_stride': 1,
    'max_val_batches': -1,
    'batches_per_epoch': 1000,
    'duplicate_as_window': False,
    'full_transformer': False,
    'pretraining': False,
    'modif_ratio': 0.5, 
    'batch_size_validation': 1024,
    'batch_size_test': 256,

    # Transformers Params 
    'd_model': 64,
    'embedding_size': 64,

    'n_heads': 8,
    'dim_ff': 256,
    'n_layers': 1,
    'latent_dim': 32,
    'q_dim': 64,
    'v_dim': 64,
    'encoding_type': EncodingTypes.NO_ENCODING,
    'normalization': True,
    'final_norm': True,

    # CNN Params:
    'use_cnn_encoder': True,
    'cnn_num_layers': 3,
    'cnn_in_channels': 1,
    'cnn_channels': 31,
    'cnn_kernel_size': 7,
    'cnn_stride_conv': 1,
    'cnn_padding': 0,
    'cnn_dilation': 1,
    'pool_kernel_size': 7,
    'pool_stride_conv': 1,
    'pool_padding': 0,
    'pool_dilation': 1, 
    'min_output_size': 64,
    'cnn_linear_size': -1,
    
    # GRU LSTM params
    'gru_hidden_size': 7,
    'gru_num_layers': 1,

    # Real CNN Params
    'conv_ker_size': 50,
    'pool_ker_size': 25,

    # Classifier_params
    'hidden_mlp': 64,
    'use_last': False,

    # Training params
    'max_duration': int(71.5 * 3600),
    'data_threshold': 0.2, # Threshold for the dataset
    'threshold': 0.5, 
    'lr': 0.0005,
    'betas': (0.9, 0.99),
    'clip': 0.5,
    'warmup_steps': 10000,
    'lr_decay': 0.99999,
    'log_every': 100,
    'save_every': 10000,
    'dropout': 0.5,
    'epochs': 4000,
    'epochs_pretrain': 30,
    'es_epochs': 100000,
    'lam': 0.2,
    'freeze_pretrained': False,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'lr_step_size': 10000,
    'lr_gamma': 1,
    'weight_decay': 0.01,

    # Pretraining data 
    'max_val_num': 3000,
    'num_training_sets': 3,
    'num_datapoints': 100000,
    'es_delta': 0.01,
    'reconstruction_dim': 64,
    'epoch_length': -1,
    'classes': 2,

    # Masking params
    'ratio_masked': 0.3,
    'ratio_replaced': 0.0,
    'ratio_kept': 1.0,

    # Finetuning data config
    # 'subjects_path': DATASET_PATH,
    # 'data_path': os.path.join(DATASET_PATH, 'dataset_classification_full_big_250_matlab_standardized_envelope_pf.txt'),
    'len_segment': 115 * 250,
    'fe': 250,
    'seed': None,

    # # Augmentation Config
    # 'jitter_scale_ratio': 1.5,
    # 'jitter_ratio': 2,
    # 'max_seg': 12,

    # # Encoder Config
    # 'input_channels': 1,
    # 'kernel_size': 8, 
    # 'stride': 1,
    # 'final_out_channels': 128,
    # 'CNNoutput_channel': 16,
    # 'temperature': 0.2,
    # 'use_cosine_similarity': True
}

SAMPLEABLE_DICT = {
    'd_model': [16, 256, 16],
    'batch_size' : [16, 64, 16],
    'seq_len': [16, 256, 16],
    'window_size': [16, 256, 16],
    'seq_stride': [16, 128, 16],
    'n_heads': [1, 10, 1],
    'dim_hidden': [128, 2048, 16],
    'n_layers': [1, 10, 1],
    'dropout': [0, 0.5, 0.1],
    'lr': [1e-6, 1e-5, 5e-6],
    'lam': [0.1, 0.7, 0.01]
}


def fill_config(config):
    """
    Takes an input config and fills in the missing values using the DEFAULT_CONFIG.
    """
    global DEFAULT_CONFIG
    for key in DEFAULT_CONFIG:
        if key not in config:
            config[key] = DEFAULT_CONFIG[key]
    return config


def validate_config(config):
    """Checks if the input config is valid.

    Args:
        config (dict): Input config

    Returns:
        Dict: The config, modified if necessary
    """
    # # Check d_model 
    # # If we are not using the cnn encoder, then d_model should be the same as window size
    # if not config['use_cnn_encoder']:
    #     config['embedding_size'] = config['window_size']
    # # If we use one hot encoding, Embedding size+seq_len
    # if config['encoding_type'] == EncodingTypes.ONE_HOT_ENCODING:
    #     config['d_model'] = config['embedding_size'] + config['seq_len']
    # elif config['d_model'] < 0:
    #     return False
    # elif config['d_model'] != config['embedding_size']:
    #     # Any other case, embedding size is the same as d_model
    #     config['embedding_size'] = config['d_model']

    # Make sure embedding_size is well set
    if config['use_cnn_encoder'] or config['duplicate_as_window']:
        # If we are using a CNN or duplicate as window and one hot encoding, embedding size is d_model/2
        if config['encoding_type'] == EncodingTypes.ONE_HOT_ENCODING:
            config['embedding_size'] = int(config['d_model'] - config['seq_len'])
        # If we are using a CNN or duplicate as window and positional encoding, embedding size is d_model
        elif config['encoding_type'] == EncodingTypes.POSITIONAL_ENCODING:
            config['embedding_size'] = config['d_model']
    # If we are using no CNN, embedding size is the window size and we check that d_model is the right dimension
    else:
        config['embedding_size'] = config['window_size']
        if config['encoding_type'] == EncodingTypes.ONE_HOT_ENCODING and config['d_model'] != (config['window_size'] + config['seq_len']):
            return False
        elif config['encoding_type'] == EncodingTypes.POSITIONAL_ENCODING and config['d_model'] != config['window_size']:
            return False

    # Check CNN params and make sure CNN params are well initialized
    if not check_valid_cnn(config): 
        return False

    # If use_cnn is True, duplicate as windows must be false
    if config['use_cnn_encoder'] and config['duplicate_as_window']:
        return False

    # Check if d_model is large enough if we want to use normalization
    if (config['normalization'] or config['final_norm']) and config['d_model'] < 16:
        return False

    return True

def sample_config_dict(exp_name, prev_exp, all_exps):
    """
    Sample a new Experiment dictionary based off of the previous experiments

    Args:
        exp_name (str): Name of the experiment
        prev_exp (dict): Dictionary of the previous experiment
        all_exps (list(dict)): List of all experiments done previously

    Returns:
        (dict, dict): Sampled dictionary and Unrounded version
    """
    config_dict = initialize_config(exp_name)
    if not validate_config(config_dict):
        raise AttributeError("Issue with your config.")
    flag_in_exps = True

    while flag_in_exps:
        noise = choices(population=[True, False], weights=[EPSILON_NOISE, 1.0 - EPSILON_NOISE])[0]
        if prev_exp == {} or noise:
            logging.debug(f"sampling random config")  # TODO: remove
            sampled_config, sample_config_unrounded = sample_once()
        else:
            logging.debug(f"sampling config near previous experiment")  # TODO: remove
            center = get_sampleable_from_config(prev_exp)
            sampled_config, sample_config_unrounded = sample_once(center=center)
        flag_in_exps = False
        for exp in all_exps:
            compared_config = get_sampleable_from_config(exp['config_dict'])
            if compare_configs(compared_config, sampled_config):
                flag_in_exps = True
                logging.debug(f"DEBUG : config already tried = {compared_config}")
                break
    
    for key in sampled_config.keys():
        config_dict[key] = sampled_config[key]

    return config_dict, sample_config_unrounded

def get_sampleable_from_config(config):
    """Gets only sampleable keys in a config dictionary

    Args:
        config (dict): Desired dictionary

    Returns:
        dict: only contains sampleable keys from config dict
    """
    sampleable = {}
    for key in SAMPLEABLE_DICT.keys():
        sampleable[key] = config[key]
    return sampleable

def compare_configs(config1, config2):
    """Compares two dictionaries based only on sampleable keys

    Args:
        config1 (dict): first dictionary to compare
        config2 (dict): second dictionary to compare

    Returns:
        Bool: True if both are the same, False otherwise
    """
    # not_compared_keys = ['exp_name', 'max_duration', 'nb_epochs_max', 'subjects_path', 'data_path']
    for key in SAMPLEABLE_DICT.keys():
        if config1[key] != config2[key]:
            return False
    return True

def clip(x, min_x, max_x):
    return max(min(x, max_x), min_x)

def sample_once(center=None, std=0.1):
    """Sample all Sampleable keys from a dictionary randomly once

    Args:
        center (dict, optional): Dictionary around which to center sampling. 
            Needs to contain all sampleable keys. Defaults to None.
        std (float, optional): Sample standard deviation. Defaults to 0.1.

    Returns:
        (dict, dict): Sampled dictionary and unrounded version of the same sample
    """
    not_valid = True
    sample, sample_unrounded = {}, {}

    while not_valid:
        for key in SAMPLEABLE_DICT.keys():
            sample[key], sample_unrounded[key] = sample_from_range(
                SAMPLEABLE_DICT[key], 
                gaussian_mean=(center[key] if center is not None else None), 
                gaussian_std_factor=std)
        
        if sample['d_model'] % sample['n_heads'] == 0:
            not_valid = False

    return sample, sample_unrounded

def sample_from_range(range_t, gaussian_mean=None, gaussian_std_factor=0.1):
    """Sample one value from a key based on a range

    Args:
        range_t (list): List of range in the format [min, max, step]
        gaussian_mean (float, optional): Gaussian center around which to sample. Defaults to None.
        gaussian_std_factor (float, optional): Standard deviation around which to sample. Defaults to 0.1.

    Returns:
        (float, float): Rounded and unrounded version of the sample
    """
    step = range_t[2]
    shift = range_t[0] % step
    min_t = round(range_t[0] / step)
    max_t = round(range_t[1] / step)
    diff_t = max_t - min_t
    gaussian_std = gaussian_std_factor * diff_t
    if gaussian_mean is None:
        res = uniform(min_t - 0.5, max_t + 0.5)  # otherwise extremum are less probable
    else:
        res = gauss(mu=gaussian_mean, sigma=gaussian_std)
        res = clip(res, min_t, max_t)
    res_unrounded = deepcopy(res) * step
    res = round(res)
    res *= step
    res += shift
    res = clip(res, range_t[0], range_t[1])
    res_unrounded = clip(res_unrounded, range_t[0], range_t[1])
    return res, res_unrounded


def check_valid_cnn(config):
    """Check for the validity of the CNN parameters and updates the necessary parameters

    Args:
        config (dict): the config to check

    Returns:
        Bool: True if config is valid, false otherwise
    """
    l_out = config['window_size']
    channels = config['cnn_in_channels']
    for _ in range(config['cnn_num_layers']):
        channels = config['cnn_channels']
        l_out = out_dim(l_out, config['cnn_padding'], config['cnn_dilation'], config['cnn_kernel_size'], config['cnn_stride_conv'])
        l_out = out_dim(l_out, config['pool_padding'], config['pool_dilation'], config['pool_kernel_size'], config['pool_stride_conv'])
    
    if l_out * channels < config['min_output_size']:
        logging.info(f"Output size of CNN {l_out * channels} is smaller than minimum {config['min_output_size']}")
        return False
    config['cnn_linear_size'] = l_out
    return True

def out_dim(window_size, padding, dilation, kernel, stride):
    return floor((window_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)