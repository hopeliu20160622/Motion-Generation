import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size):
        super().__init__()

        self.motion_embed = input_dim
        self.lstm_size = hidden_dim
        self.num_layers = 1
        self.batch_size = batch_size
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
        self.l1_loss = nn.L1Loss()
    
    def forward(self, x, x_lengths):
        h0 = torch.randn(self.num_layers,x.shape[0],self.lstm_size)
        c0 = torch.randn(self.num_layers,x.shape[0],self.lstm_size)

        x = self.encoder1(x)
        x = self.prelu(x)
        x = self.encoder2(x)

        x = pack_padded_sequence(x, x_lengths, batch_first=True)
        out, _ = self.lstm(x, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)
        
        out = self.decoder2(out)
        out = self.prelu(out)
        out = self.decoder1(out)
        return out

    def loss(self, preds, targets, x_lengths):
        # TODO: doc Why calculate loss conditioned on lengths. It will prevent preferring short seq
        total_loss = 0
        for batch_id, length in enumerate(x_lengths):
            loss_single = self.l1_loss(preds[batch_id, :length,:], targets[batch_id, :length, :])
            total_loss += loss_single
        mean_loss = total_loss / len(x_lengths)
        return mean_loss
