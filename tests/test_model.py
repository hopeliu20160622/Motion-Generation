from mg.model.lstm import LSTMModel
import torch
import numpy as np

def test_model():
    mat = np.load('npz_data/quaternion/F01A0V1.npz')
    input_mat = np.expand_dims(np.concatenate([mat['pos'], mat['rot']], axis=1), axis=0)
    model = LSTMModel()
    h0 = torch.randn((1,1,128))
    c0 = torch.randn((1,1,128))
    input_mat = torch.Tensor(input_mat)
    out, (hn, cn) = model(input_mat, (h0, c0))