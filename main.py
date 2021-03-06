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

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_files, test_files = extract_train_test_path(meta_file='data/emotion-mocap/file-info.csv', filename='filename', target='emotion', test_size=0.1)
    npz = convert_bvh_path_to_npz(train_files)
    X_padded, y_padded, seq_lengths = make_padded_batch(npz)

    dataset = MotionDataset(X_padded, y_padded, seq_lengths)
    batch_size = 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_dim = 63 # HARD CODING
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
            x_padded = x_padded[:,:,:63].to(device)
            output = model(x_padded, seq_lengths) # double check s_seq_lengths valied
            pred = output[:,:,:63] #XYZ ONLY
            target = y_padded[:,:,:63].to(device) #XYZ ONLY
            loss = model.loss(pred, target, seq_lengths)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        loss_hist.append(loss.item())
    
    print("Training Complete.")
    plt.plot(loss_hist)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig('loss_graph.png')
    save_path = f"saved_weights_{num_epochs}"
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()