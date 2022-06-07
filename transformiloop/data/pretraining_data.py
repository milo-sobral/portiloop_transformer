import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, X, Y, input_noise):
        self.X = X
        self.Y = Y
        self.input_noise = input_noise
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _x = self.X[index]
        _y = self.Y[index]

        if self.input_noise:
            _x = _x + torch.tensor(np.random.normal(0, 0.3, size=_x.shape), dtype=torch.float32)

        return _x.to(device), _y.to(device), _x.to(device)


def create_sequences(input_data, seq_len, output_window):
    '''
    Generate sequences of input and output for the model to learn
    '''
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

def prepare_from_file(data, 
                      batch_size, 
                      num_train, 
                      input_window, 
                      output_window,
                      recreate=False,
                      train_percentage=0.7, 
):
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