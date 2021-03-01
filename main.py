import pandas as pd
import torch
from torch import nn

# Split Train Test
## Load file Meta {file: PyMO}
## Split Train Test after shuffling
## Now data will be split into train_path, tarin_BVHM, test_path, test_BVHM


# Train Model
## Make Pipeline to Train it with GRU first
##

# Inference on Test set
## Visualization
## Performance Metric


rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output)