import torch.nn as nn
import torch


class TFC(nn.Module): # Frequency domain encoder
    def __init__(self, config):
        super(TFC, self).__init__()

        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(config['input_channels'], 32, kernel_size=config['kernel_size'],
                      stride=config['stride'], bias=False, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(config['dropout'])
        )

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, config['final_out_channels'], kernel_size=8, stride=1, bias=False, padding='same'),
            nn.BatchNorm1d(config['final_out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        self.projector_t = nn.Sequential(
            nn.Linear(config['final_out_channels'] * (config['window_size'] // 8), config['d_model'] * 2),
            nn.BatchNorm1d(config['d_model'] * 2),
            nn.ReLU(),
            nn.Linear(config['d_model'] * 2, config['d_model'] // 2)
        )

        self.conv_block1_f = nn.Sequential(
            nn.Conv1d(config['input_channels'], 32, kernel_size=config['kernel_size'],
                      stride=config['stride'], bias=False, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(config['dropout'])
        )

        self.conv_block2_f = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )

        self.conv_block3_f = nn.Sequential(
            nn.Conv1d(64, config['final_out_channels'], kernel_size=8, stride=1, bias=False, padding='same'),
            nn.BatchNorm1d(config['final_out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        self.projector_f = nn.Sequential(
            nn.Linear(config['final_out_channels'] * (config['window_size'] // 8), config['d_model'] * 2),
            nn.BatchNorm1d(config['d_model'] * 2),
            nn.ReLU(),
            nn.Linear(config['d_model'] * 2, config['d_model'] // 2)
        )


    def forward(self, x_in_t, x_in_f):

        """Time-based Contrastive Encoder"""
        x = self.conv_block1_t(x_in_t)
        x = self.conv_block2_t(x)
        
        x = self.conv_block3_t(x)

        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.conv_block1_f(x_in_f)
        f = self.conv_block2_f(f)
        f = self.conv_block3_f(f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq

    