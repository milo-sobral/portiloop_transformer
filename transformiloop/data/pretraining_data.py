import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, 
    X: list, 
    Y: list, 
    input_noise: bool):
        """Dataset Wrapper for Sequence to Sequence pretraining

        Args:
            X (list): List of all input sequences
            Y (list): List of all desired output sequences
            input_noise (bool): Booelan to determine if we want to add noise to input

        Raises:
            Exception: Raises exception if the length of X does not match length of Y
        """
        self.X = X
        self.Y = Y
        self.input_noise = input_noise
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        """_summary_

        Returns:
            int: Length of the dataset
        """
        return len(self.X)

    def __getitem__(self, index):
        """Get one item from the dataset

        Args:
            index (int): index of the desired datapoint

        Returns:
            a tuple (x, y, x): x is input sequence, y is desired prediction sequence.
        """
        _x = self.X[index]
        _y = self.Y[index]

        if self.input_noise:
            _x = _x + torch.tensor(np.random.normal(0, 0.3, size=_x.shape), dtype=torch.float32)

        return _x.to(device), _y.to(device), _x.to(device)


def create_sequences(input_data: list, 
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

def prepare_from_file(data: list, 
                      batch_size: int, 
                      num_train: int, 
                      input_window: int, 
                      output_window: int,
                      recreate=False: int,
                      train_percentage=0.7: float,):
    """_summary_

    Args:
        data (list): _description_
        batch_size (int): _description_
        num_train (int): _description_
        input_window (int): _description_
        output_window (int): _description_
        recreate (_type_, optional): _description_. Defaults to False:int.
        train_percentage (_type_, optional): _description_. Defaults to 0.7:float.

    Returns:
        _type_: _description_
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
        SequenceDataset(train_sequence_x, train_sequence_y, input_noise=False), 
        batch_size=batch_size, 
        shuffle=True)
    val_loader = DataLoader(
        SequenceDataset(val_sequence_x, val_sequence_y, input_noise=False), 
        batch_size=batch_size, 
        shuffle=True)
    return train_loader, val_loader