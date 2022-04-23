import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FCN(nn.Module):
    def __init__(self, inp_size, out_size, n_hidden):
        super(FCN, self).__init__()
        self.name = 'FCN'
        self.n_hidden = n_hidden
        self.n_inp = inp_size
        self.n_out = out_size
        self.fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_inp, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_out), nn.Sigmoid(),
        )

    def forward(self, inp):
        out = self.fcn(inp)
        return out