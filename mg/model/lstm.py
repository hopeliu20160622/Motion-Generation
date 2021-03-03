import torch
from torch import nn

class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.motion_embed = 63 + 84
        self.lstm_size = 128
        self.encoder1 = nn.Linear(self.motion_embed, 256)
        self.prelu = nn.PReLU()
        self.encoder2 = nn.Linear(256, 128)
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            batch_first=True
        )
        self.decoder2 = nn.Linear(self.lstm_size, 256)
        self.decoder1 = nn.Linear(256, self.motion_embed)
    
    def forward(self, x):
        h0 = torch.randn((1,1,128))
        c0 = torch.randn((1,1,128))

        embed = self.encoder1(x)
        embed = self.prelu(embed)
        embed = self.encoder2(embed)

        output, _state = self.lstm(embed, (h0, c0))

        out = self.decoder2(output)
        out = self.prelu(out)
        out = self.decoder1(out)
        return out
