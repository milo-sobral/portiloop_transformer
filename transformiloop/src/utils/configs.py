import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_default_config():
    return DEFAULT_CONFIG

training_config = {
    # Training params
    'lr': 1e-5,
    'betas': (0.9, 0.99),
    'clip': 5,
    'log_every': 100,
    'dropout': 0.1,
    'epochs': 10,
    'device': device,
}

transformer_config = {
    'd_model': 256,
    'n_heads': 8,
    'dim_hidden': 1024,
    'n_layers': 6,
    'latent_dim': 32,
}

pretraining_data_config = {
    'threshold': 0.5,
    'max_val_num': 3000,
    'num_training_sets': 3,
    'num_datapoints': 100000,
}


MODA_data_config = {
    'subjects_path': '/content/drive/Othercomputers/My Laptop/_Data/Portiloop/portiloop_software/dataset',
    'data_path': '/content/drive/Othercomputers/My Laptop/_Data/Portiloop/portiloop_software/dataset/dataset_classification_full_big_250_matlab_standardized_envelope_pf.txt',
    'window_size': 256,
    'seq_len': 64,
    'seq_stride': 7,
    'network_stride': 1,
    'len_segment': 115 * 250,
    'fe': 250,
    'threshold': 0.5,
    'batch_size': 64,
    'training_batches': 10000,
    'validation_batches': 5000,
    'seed': None
}

augmentation_config = {
    'jitter_scale_ratio': 1.5,
    'jitter_ratio': 2,
    'max_seg': 12
}

encoder_config = {
    'input_channels': 1,
    'kernel_size': 8, 
    'stride': 1,
    'final_out_channels': 128,
    'CNNoutput_channel': 66,
    'dropout': 0.35,
    'd_model': 256,
}

DEFAULT_CONFIG = {
    # Data params
    'batch_size' : 64,
    'seq_len': 256,
    'out_seq_len': 64,

    # Subconfigs
    'training': training_config,
    'transformer_config': transformer_config,
    'pretraining_data_config': pretraining_data_config,
    'MODA_data_config': MODA_data_config,
    'augmentation_config': augmentation_config,
    'encoder_config': encoder_config,

    # TODO: figure out where to put those
    'task': 'both',
    'mode': 'classification',
    'recreate': None,
    'temperature': 0.2,
    'use_cosine_similarity': True
}

