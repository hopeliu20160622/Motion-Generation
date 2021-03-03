import torch
from torch import nn

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.motion_embed = input_dim
        self.lstm_size = hidden_dim
        self.encoder1 = nn.Linear(self.motion_embed, 256)
        self.prelu = nn.PReLU()
        self.encoder2 = nn.Linear(256, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            batch_first=True
        )
        self.decoder2 = nn.Linear(self.lstm_size, 256)
        self.decoder1 = nn.Linear(256, self.motion_embed)
    
    def forward(self, x):
        # h0 = torch.randn((1,1,self.lstm_size))
        # c0 = torch.randn((1,1,self.lstm_size))

        embed = self.encoder1(x)
        embed = self.prelu(embed)
        embed = self.encoder2(embed)

        output, _state = self.lstm(embed)

        out = self.decoder2(output)
        out = self.prelu(out)
        out = self.decoder1(out)
        return out
