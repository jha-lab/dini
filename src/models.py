import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class FCN(nn.Module):
    def __init__(self, inp_size, out_size, n_hidden, mc_dropout=False):
        super(FCN, self).__init__()
        self.name = 'FCN'
        self.n_hidden = n_hidden
        self.n_inp = inp_size
        self.n_out = out_size
        self.fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_inp, self.n_hidden), nn.LeakyReLU(True),
            nn.Dropout(0.1 if mc_dropout else 0),
            nn.Linear(self.n_hidden, self.n_out), nn.Sigmoid(),
        )

    def forward(self, inp, out):
        out2 = self.fcn(inp)
        return inp, out2

class FCN2(nn.Module):
    def __init__(self, inp_size, out_size, n_hidden, mc_dropout=False):
        super(FCN2, self).__init__()
        self.name = 'FCN2'
        self.n_hidden = n_hidden
        self.n_inp = inp_size
        self.n_out = out_size
        self.fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_inp, self.n_hidden), nn.LeakyReLU(True),
            nn.Dropout(0.1 if mc_dropout else 0),
            nn.Linear(self.n_hidden, self.n_out), nn.Sigmoid(),
        )
        self.fcn_reverse = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_out, self.n_hidden), nn.LeakyReLU(True),
            nn.Dropout(0.1 if mc_dropout else 0),
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

class TXF2(nn.Module):
    def __init__(self, inp_size, out_size, n_hidden, use_pos_emb=False):
        super(TXF2, self).__init__()
        self.name = 'TXF2'
        self.hidden_size = n_hidden
        self.inp_size = inp_size
        self.out_size = out_size
        self.pos_emb = nn.Linear(self.inp_size, self.inp_size)
        self.pos_emb_reverse = nn.Linear(self.out_size, self.out_size)
        self.use_pos_emb = use_pos_emb
        self.txf = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=self.inp_size,
                nhead=1,
                dim_feedforward=self.hidden_size,
                batch_first=True),
            nn.Linear(self.inp_size, self.out_size)
            )
        self.txf_reverse = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=self.out_size,
                nhead=1,
                dim_feedforward=self.hidden_size,
                batch_first=True),
            nn.Linear(self.out_size, self.inp_size)
            )

    def forward(self, inp, out):
        if self.use_pos_emb:
            inp = inp + self.pos_emb(inp)
            out = out + self.pos_emb_reverse(out)
        out2 = self.txf(inp)
        inp2 = self.txf_reverse(out)
        return inp2, out2
