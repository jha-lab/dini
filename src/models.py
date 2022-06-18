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
            nn.Linear(self.n_inp, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_out), nn.Sigmoid(),
        )

    def forward(self, inp, out):
        out2 = self.fcn(inp)
        return inp, out2

class FCN2(nn.Module):
    def __init__(self, inp_size, out_size, n_hidden):
        super(FCN2, self).__init__()
        self.name = 'FCN2'
        self.n_hidden = n_hidden
        self.n_inp = inp_size
        self.n_out = out_size
        self.fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_inp, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_out), nn.Sigmoid(),
        )
        self.fcn_reverse = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_out, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_inp), nn.Sigmoid(),
        )

    def forward(self, inp, out):
        out2 = 2 * self.fcn(inp) - 0.5
        inp2 = 2 * self.fcn_reverse(out) - 0.5
        return inp2, out2

class LSTM2(nn.Module):
    def __init__(self, inp_size, out_size, n_hidden):
        super(LSTM2, self).__init__()
        self.name = 'LSTM2'
        self.hidden_size = n_hidden
        self.inp_size = inp_size
        self.out_size = out_size
        self.lstm = nn.LSTM(input_size=self.inp_size,
            hidden_size=self.hidden_size,
            proj_size=self.out_size,
            # bidirectional=True,
            batch_first=True)
        self.lstm_reverse = nn.LSTM(input_size=self.out_size,
            hidden_size=self.hidden_size,
            proj_size=self.inp_size,
            # bidirectional=True,
            batch_first=True)

    def forward(self, inp, out):
        out2, _ = self.lstm(inp)
        inp2, _ = self.lstm_reverse(out)
        return inp2, out2
