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

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_files, test_files = extract_train_test_path(meta_file='data/emotion-mocap/file-info.csv', filename='filename', target='emotion', test_size=0.1)
    test_npz = convert_bvh_path_to_npz(test_files)
    X_padded, y_padded, seq_lengths = make_padded_batch(test_npz[:13])

    dataset = MotionDataset(X_padded, y_padded, seq_lengths)
    batch_size = 10
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_dim = 3
    hidden_dim = 128
    model = LSTMModel(input_dim, hidden_dim, batch_size=batch_size, device=device)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters())

    # test output w/o training
    num_epochs = 5
    loss_hist = []
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        for x_padded, y_padded, seq_lengths in loader:
            optimizer.zero_grad()
            x_padded = x_padded[:,:,:3].to(device)
            output = model(x_padded, seq_lengths) # double check s_seq_lengths valied
            pred = output[:,:,:3] #XYZ ONLY
            target = y_padded[:,:,:3].to(device) #XYZ ONLY
            loss = model.loss(pred, target, seq_lengths)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
    
    print("Training Complete.")
    save_path = "saved_weights"
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()