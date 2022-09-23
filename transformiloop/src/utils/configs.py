import logging
from copy import deepcopy
from math import floor
from random import choices, uniform, gauss
import pathlib
import os
import torch

EPSILON_NOISE = 0.25 # Proportion of samples which are fully random

def get_default_config(name):
    DEFAULT_CONFIG['exp_name'] = name
    return DEFAULT_CONFIG


DEFAULT_CONFIG = {
    # Data params
    'batch_size' : 32,
    'seq_len': 512,
    'window_size': 32,
    'seq_stride': 1,
    'val_batch_size': 350,
    'val_dividing_factor': 20,
    'test_dividing_factor': 1,
    'batches_per_epoch': 500,
    'duplicate_as_window': True,

    # Transformers Params 
    'd_model': 32,
    'n_heads': 8,
    'dim_ff': 256,
    'n_layers': 6,
    'latent_dim': 32,
    'q_dim': 32,
    'v_dim': 32,

    # Training params
    'max_duration': int(71.5 * 3600),
    'threshold': 0.5, 
    'lr': 7e-4,
    'betas': (0.9, 0.99),
    'clip': 5,
    'warmup_steps': 4000,
    'lr_decay': 0.9999,
    'log_every': 50,
    'dropout': 0.0,
    'epochs': 400,
    'epochs_pretrain': 30,
    'es_epochs': 100,
    'lam': 0.2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'final_norm': False,

    # Pretraining data 
    'max_val_num': 3000,
    'num_training_sets': 3,
    'num_datapoints': 100000,
    'es_delta': 0.01,

    # Finetuning data config
    # 'subjects_path': DATASET_PATH,
    # 'data_path': os.path.join(DATASET_PATH, 'dataset_classification_full_big_250_matlab_standardized_envelope_pf.txt'),
    'len_segment': 115 * 250,
    'fe': 250,
    'validation_batches': 100000,
    'seed': None,

    # Augmentation Config
    'jitter_scale_ratio': 1.5,
    'jitter_ratio': 2,
    'max_seg': 12,

    # Encoder Config
    'input_channels': 1,
    'kernel_size': 8, 
    'stride': 1,
    'final_out_channels': 128,
    'CNNoutput_channel': 16,
    'temperature': 0.2,
    'use_cosine_similarity': True
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
    config_dict = get_default_config(exp_name)
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

