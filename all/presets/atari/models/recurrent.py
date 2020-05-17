import torch
from all import nn

class RecurrentFeatures(nn.Module):
    def __init__(self, conv, lstm, num_layers):
        super().__init__()
        self.conv = conv
        self.lstm = lstm
        self.num_layers = num_layers
        self.device = next(conv.parameters()).device

    def forward(self, x):
        n_states = x.shape[0]
        history_length = x.shape[1]
        # move everything into batch dimension
        x = x.view((-1, 1) + x.shape[2:])
        # computer forward pass of conv model
        x = self.conv(x)
        # restack conv features corresponding to the same history
        x = x.view((n_states, history_length, -1))
        # get initial hidden state
        hidden_state = torch.zeros((self.num_layers, n_states, x.shape[-1]), device=self.device)
        cell_state = torch.zeros((self.num_layers, n_states, x.shape[-1]), device=self.device)
        # let pytorch loop through all inputs at once
        out, _ = self.lstm(x, (hidden_state, cell_state))
        # return just the output, not the hidden state
        return out[:, -1, :]

    def to(self, device):
        super().to(device)
        self.device = device
        return self

def recurrent_features(num_layers=2):
    conv = nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(1, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
    )
    lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True)
    return RecurrentFeatures(conv, lstm, num_layers)
