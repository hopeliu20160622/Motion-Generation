from mg.model.lstm import LSTMModel
from mg.data.util import make_padded_batch, extract_train_test_path, convert_bvh_path_to_npz
from mg.data.preprocess import MotionDataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, test_files = extract_train_test_path(meta_file='data/emotion-mocap/file-info.csv', filename='filename', target='emotion', test_size=0.1)
    npz = convert_bvh_path_to_npz(test_files)
    pos = np.load(npz[50])['pos']
    batch_size = 64

    X_padded = torch.Tensor(pos[np.newaxis,:,:])
    for_viz = X_padded[0].numpy()

    input_dim = 63
    hidden_dim = 128
    model = LSTMModel(input_dim, hidden_dim, batch_size=batch_size, device=device)
    model.load_state_dict(torch.load('saved_weights/position_only/saved_weights_100', map_location=device))
    model.eval()
    model.to(device)
    
    # Sequential generation, last 30 frames continue
    input_frames = 100
    generate_frames = 100

    generated_collated = []
    frames = X_padded[:,:input_frames,:]
    for _ in range(generate_frames):
        pred = model(frames, [input_frames])
        seen = frames[0][1:] #39 X 3
        generated = pred[0][-1:,:]
        frames = torch.unsqueeze(torch.cat((seen,generated), dim=0), 0)
        generated_collated.append(generated)
    generated_collated = torch.unsqueeze(torch.cat(generated_collated), 0)

    # visualize
    original_frames = X_padded[0].numpy()
    print(original_frames)

if __name__ == '__main__':
    inference()