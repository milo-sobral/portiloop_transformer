import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PretrainingDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, sequences, augmentation_config):
        super(PretrainingDataset, self).__init__()
        # shuffle
        np.random.shuffle(sequences)
        # X_train, y_train = zip(*data)
        # X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)
        X_train = torch.stack(list(sequences), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        # self.x_data_f = self.x_data_f[:, :, 1:] # not a problem.

        self.len = X_train.shape[0]
        """Augmentation"""

        self.aug1 = DataTransform_TD(self.x_data, augmentation_config)
        self.aug1_f = DataTransform_FD(self.x_data_f) # [7360, 1, 90]

    def __getitem__(self, index):
        return self.x_data[index], self.aug1[index], self.x_data_f[index], self.aug1_f[index]

    def __len__(self):
        return self.len


def create_sequences(
    input_data: list, 
    seq_len:int, 
    output_window:int):
    """ Generate sequences of input and output to train the model on prediction tasks.

    Args:
        input_data (list): List of datapoint in pretraining dataset
        seq_len (int): Desired input sequence length
        output_window (int): Desired output sequence length for prediction

    Returns:
        tuple (x, y): x is a list of tensors of size seq_len and y is a list of tensors of size output_window 
    """
    seqs_x = []
    seqs_y = []
    for i in range(len(input_data) - seq_len - output_window):
        train_seq = input_data[i:i+seq_len] # Input of the model
        train_label = input_data[i+seq_len:i+seq_len+output_window]
        seqs_x.append(train_seq)
        seqs_y.append(train_label)

    array_x = np.stack(seqs_x)
    array_y = np.stack(seqs_y)
    return torch.FloatTensor(array_x), torch.FloatTensor(array_y)

def prepare_from_file(
    data: list, 
    batch_size: int, 
    num_train: int, 
    input_window: int, 
    output_window: int,
    device, 
    train_percentage=0.7):
    """Prepares a training and a validation dataset from scratch

    Args:
        data (list): _description_
        batch_size (int): _description_
        num_train (int): _description_
        input_window (int): _description_
        output_window (int): _description_
        train_percentage (float, optional): _description_. Defaults to 0.7.

    Returns:
        x: Torch DataLoader with training split
        y: Torch loader with validation split
    """
    if num_train is None:
        samples = int(len(data) * train_percentage)
        end = len(data)
    else:
        samples = num_train
        end = int((1-train_percentage) * num_train) + num_train
    train_data = data[:samples]
    val_data = data[samples:end]

    train_sequence_x, train_sequence_y = create_sequences(train_data, input_window, output_window)
    val_sequence_x, val_sequence_y = create_sequences(val_data, input_window, output_window)

    train_loader = DataLoader(
        SequenceDataset(train_sequence_x, train_sequence_y, device, input_noise=False), 
        batch_size=batch_size, 
        shuffle=True)
    val_loader = DataLoader(
        SequenceDataset(val_sequence_x, val_sequence_y, device, input_noise=False), 
        batch_size=batch_size, 
        shuffle=True)
    return train_loader, val_loader