from mg.model.lstm import LSTMModel
from mg.data.util import make_seq_list
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from torch import nn
import torch.optim as optim

def test_model_train():
    files = ['npz_data/quaternion/F01A0V1.npz', 'npz_data/quaternion/M11SU0V1.npz', 'npz_data/quaternion/M11SU4V1.npz',
              'npz_data/quaternion/M11H2V1.npz', 'npz_data/quaternion/M11D3V1.npz']
    mats = make_seq_list(files)
    sorted_mats = sorted(mats, key=lambda x: x.shape[0], reverse=True) # sort by seq length, descending order
    input_mat = pad_sequence(sorted_mats, batch_first=True) 
    
    X = input_mat[:, :-1, :]
    Y = input_mat[:, 1:, :]

    input_dim = X.shape[2]
    hidden_dim = 128
    model = LSTMModel(input_dim, hidden_dim)
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(params=model.parameters())
    # test output w/o training
    num_epochs = 10
    loss_hist = []
    for _ in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        pred = output
        target = Y
        assert pred.shape == target.shape
        loss = loss_function(pred, target)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
    assert loss_hist[0] > loss_hist[-1]


    



## Test Train batch
## Visualize 3d xyz test (test purpose)