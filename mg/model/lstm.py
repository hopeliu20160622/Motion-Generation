import torch
from torch import nn

class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.motion_embed = 63 + 84
        self.lstm_size = 128
        self.encoder = nn.Linear(self.motion_embed, 128)
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            batch_first=True
        )
        self.decoder = nn.Linear(self.lstm_size, self.motion_embed)
    
    def forward(self, x, prev_state):
        embed = self.encoder(x)
        output, state = self.lstm(embed, prev_state)
        out = self.decoder(output)
        return out, state 
