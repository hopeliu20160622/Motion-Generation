from mg.model.lstm import LSTMModel
from mg.data.util import make_seq_list, split_seq_train_test
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from torch import nn
import torch.optim as optim

def test_model_train():
    files = ['npz_data/quaternion/F01A0V1.npz', 'npz_data/quaternion/M11SU0V1.npz', 'npz_data/quaternion/M11SU4V1.npz',
              'npz_data/quaternion/M11H2V1.npz', 'npz_data/quaternion/M11D3V1.npz']
    mats = make_seq_list(files)
    x_seq_lengths = sorted([mat.shape[0] - 1 for mat in mats], reverse=True)
    sorted_mats = sorted(mats, key=lambda x: x.shape[0], reverse=True) # sort by seq length, descending order
    
    x, y = split_seq_train_test(sorted_mats)
    x_padded = pad_sequence(x, batch_first=True)
    y_padded = pad_sequence(y, batch_first=True)

    x_padded = x_padded[:,:,:3] #XYZ ONLY
    y_padded = y_padded[:,:,:3] #XYZ ONLY

    input_dim = x_padded.shape[2]
    hidden_dim = 128

    model = LSTMModel(input_dim, hidden_dim, batch_size=len(files))
    optimizer = optim.Adam(params=model.parameters())

    # test output w/o training
    num_epochs = 5
    loss_hist = []
    for _ in range(num_epochs):
        optimizer.zero_grad()
        output = model(x_padded, x_seq_lengths) # double check s_seq_lengths valied
        pred = output[:,:,:3] #XYZ ONLY
        target = y_padded[:,:,:3] #XYZ ONLY
        assert pred.shape == target.shape
        loss = model.loss(pred, target, x_seq_lengths)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
    assert loss_hist[0] > loss_hist[-1]

## Visualize 3d xyz test (test purpose)