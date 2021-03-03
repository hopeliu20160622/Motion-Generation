from mg.model.lstm import LSTMModel
import torch
import numpy as np
from torch import nn
import torch.optim as optim

def test_model_train():
    mat = np.load('npz_data/quaternion/F01A0V1.npz')
    input_mat = np.expand_dims(np.concatenate([mat['pos'], mat['rot']], axis=1), axis=0)
    input_mat = torch.Tensor(input_mat)
    
    X = input_mat[:, :-1, :]
    Y = input_mat[:, 1:, :]

    model = LSTMModel()
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(params=model.parameters())
    # test output w/o training
    num_epochs = 10
    loss_hist = []
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        pred = output.view(-1, output.size(2))
        target = Y.view(-1, Y.size(2))
        assert pred.shape == target.shape
        loss = loss_function(pred, target)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
    assert loss_hist[0] > loss_hist[-1]


