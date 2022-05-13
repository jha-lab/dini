import numpy as np
from src.utils import *
from src.folderconstants import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm


# GAIN implementation from: https://github.com/mertyg/gain-pytorch
class GAINGenerator(nn.Module):
    def __init__(self, dim, args, transform_params):
        super(GAINGenerator, self).__init__()
        self.args = args
        self.h_dim = dim
        norm_min = transform_params["min"]
        norm_max = transform_params["max"]
        self.register_buffer("norm_min", norm_min)
        self.register_buffer("norm_max", norm_max + 1e-6)
        self.model = None
        self.init_network(dim)
        self.seed_sampler = torch.distributions.Uniform(low=0, high=0.01)

    def init_network(self, dim):
        generator = [nn.Linear(dim * 2, dim), nn.ReLU()]
        generator.extend([nn.Linear(dim, dim), nn.ReLU()])
        generator.extend([nn.Linear(dim, dim), nn.Sigmoid()])
        for layer in generator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.model = nn.Sequential(*generator)

    def normalizer(self, inp, mode="normalize"):
        if mode == "normalize":
            inp_norm = inp - self.norm_min
            inp_norm = inp_norm / self.norm_max

        elif mode == "renormalize":
            inp_norm = inp * self.norm_max
            inp_norm = inp_norm + self.norm_min

        else:
            raise NotImplementedError()

        return inp_norm

    def forward(self, data, mask):
        # MASK: gives you non-nans
        data_norm = self.normalizer(data)
        data_norm[mask == 0] = 0.
        z = self.seed_sampler.sample([data_norm.shape[0], self.h_dim])
        random_combined = mask * data_norm + torch.logical_not(mask) * z
        sample = self.model(torch.cat([random_combined, mask], dim=1))
        x_hat = random_combined * mask + sample * (1-mask)
        return sample, random_combined, x_hat


class GAINDiscriminator(nn.Module):
    def __init__(self, dim, label_dim, args):
        super(GAINDiscriminator, self).__init__()
        self.args = args
        self.hint_rate = torch.tensor(0.9)
        self.h_dim = dim
        self.discriminator = None
        self.init_network(dim)
        self.uniform = torch.distributions.Uniform(low=0, high=1.)

    def init_network(self, dim):
        discriminator = [nn.Linear(dim * 2, dim), nn.ReLU()]
        discriminator.extend([nn.Linear(dim, dim), nn.ReLU()])
        discriminator.extend([nn.Linear(dim, dim), nn.Sigmoid()])
        for layer in discriminator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.discriminator = nn.Sequential(*discriminator)

    def forward(self, x_hat, mask):
        hint = (self.uniform.sample([mask.shape[0], self.h_dim]) < self.hint_rate).float()
        hint = mask * hint
        inp = torch.cat([x_hat, hint], dim=1)
        return self.discriminator(inp)

class GAINTrainer:

    def __init__(self, dim, label_dim, transform_params, args, load_path=None):
        self.dim = dim
        self.label_dim = label_dim
        self.discriminator = GAINDiscriminator(self.dim, self.label_dim, args)
        self.generator = GAINGenerator(self.dim, args, transform_params)
        self.batch_size = 128
        self.learning_rate = 0.001
        self.alpha = 100
        self.d_optimizer = optim.Adam(params=self.discriminator.parameters(), lr=self.learning_rate)
        self.g_optimizer = optim.Adam(params=self.generator.parameters(), lr=self.learning_rate)
        self.epoch = 0
        self.train_history = {"D_loss": [], "G_loss": [], "MSE_loss": []}
        self.eval_history = {}
        self.args = args
        self.result_dir = load_path if load_path else f"./results/GAIN/{time.strftime('%Y%m%d-%H%M%S')}"

    def discriminator_loss(self, mask, d_prob):
        d_loss = -torch.mean(mask * torch.log(d_prob+1e-8) + (1-mask) * torch.log(1-d_prob + 1e-8))
        return d_loss

    def generator_loss(self, mask, d_prob, random_combined, sample):
        g_loss = -torch.mean((1-mask) * torch.log(d_prob + 1e-8))
        mse_loss = torch.mean(torch.pow((mask * random_combined - mask*sample), 2)) / torch.mean(mask)
        return g_loss, mse_loss

    def save_checkpoint(self):
        ckpt_file = os.path.join(self.result_dir, "checkpoint")
        state_dict = dict()
        state_dict["generator"] = self.generator.state_dict()
        state_dict["discriminator"] = self.discriminator.state_dict()
        state_dict["d_optimizer"] = self.d_optimizer.state_dict()
        state_dict["g_optimizer"] = self.g_optimizer.state_dict()
        state_dict["epoch"] = self.epoch
        torch.save(state_dict, ckpt_file)

    def train_step(self, loader):
        self.discriminator.train()
        self.generator.train()
        device = torch.device('cpu')
        b_loader = loader
        for _, x_batch, _, m_batch in b_loader:
            x_batch, m_batch = x_batch.to(device), m_batch.to(device)
            self.g_optimizer.zero_grad()
            sample, random_combined, x_hat = self.generator(x_batch, m_batch)
            G_loss, mse_loss = self.generator_loss(m_batch, self.discriminator(x_hat, m_batch), random_combined, sample)
            generator_loss = G_loss + self.alpha * mse_loss
            generator_loss.backward()
            self.g_optimizer.step()

            self.d_optimizer.zero_grad()
            D_prob = self.discriminator(x_hat.detach(), m_batch)
            D_loss = self.discriminator_loss(m_batch, D_prob)
            D_loss.backward()
            self.d_optimizer.step()

            N = x_batch.shape[0]

    def rounding(self, tensor, masked_data):
        _, dim = tensor.shape
        rounded_data = tensor.clone()

        for i in range(dim):
            temp = masked_data[~torch.isnan(masked_data[:, i]), i]
            if len(torch.unique(temp)) < 20:
                rounded_data[:, i] = torch.round(rounded_data[:, i])
        return rounded_data

    def eval_model(self, loader, mode):
        device = torch.device('cpu')
        self.discriminator.eval()
        self.generator.eval()
        all_imputed = []
        all_orig = []
        all_mask = []
        all_data = []
        for x_original, x_batch, _, m_batch in loader:
            x_batch, m_batch, x_original = x_batch.to(device), m_batch.to(device), x_original.to(device)
            sample, random_combined, x_hat = self.generator(x_batch, m_batch)
            x_batch[m_batch == 0] = 0.
            imputed_data = m_batch * self.generator.normalizer(x_batch) + (1-m_batch) * sample
            all_imputed.append(imputed_data.detach())
            all_orig.append(x_original)
            all_mask.append(m_batch)
            all_data.append(x_batch)

        imputed_data = torch.cat(all_imputed, dim=0)
        x_original = torch.cat(all_orig, dim=0)
        mask = torch.cat(all_mask, dim=0)
        x = torch.cat(all_data, dim=0)
        imputed_data = self.generator.normalizer(imputed_data, mode="renormalize")
        x[mask == 0] = np.nan
        imputed_data = self.rounding(imputed_data, x)
        imputed_data = self.generator.normalizer(imputed_data)
        x_original = self.generator.normalizer(x_original)

        sq_error = torch.sum(((1 - mask) * x_original - (1 - mask) * imputed_data) ** 2)
        rmse = torch.sqrt(sq_error / ((1-mask).sum())).detach().cpu().item()
        return {mode+"-RMSE": rmse}, imputed_data

    def log_results(self, res_dict):
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        config = os.path.join(self.result_dir, "config.json")
        with open(config, 'w') as fp:
            json.dump(vars(self.args), fp)

        perf_file = os.path.join(self.result_dir, "performance.json")
        with open(perf_file, "w") as fp:
            json.dump(res_dict, fp)

    def train_model(self, train_loader, test_loader):
        device = torch.device('cpu')
        self.discriminator.to(device)
        self.generator.to(device)

        t_loader = tqdm(range(100), ncols=80)

        for i in t_loader:
            self.train_step(train_loader)
            self.epoch += 1

            desc = list([f"Epoch: {self.epoch}"])
            desc = " ".join(desc)
            t_loader.set_description(desc)

            if (self.epoch + 1) % 20 == 0:
                train_eval, imputed_data = self.eval_model(train_loader, mode="train")
                test_eval, imputed_data = self.eval_model(test_loader, mode="test")
                json.dumps(train_eval)
                json.dumps(test_eval)

        train_eval, imputed_data = self.eval_model(train_loader, mode="train")
        test_eval, imputed_data = self.eval_model(test_loader, mode="test")
        perf_dict = train_eval
        perf_dict.update(test_eval)
        # self.log_results(perf_dict)
        # self.save_checkpoint()
        return perf_dict, imputed_data

def load_data(dataset):
    inp = torch.tensor(np.load(f'{output_folder}/{dataset}/inp.npy'), dtype=torch.float)
    out = torch.tensor(np.load(f'{output_folder}/{dataset}/out.npy'), dtype=torch.float)
    inp_c = torch.tensor(np.load(f'{output_folder}/{dataset}/inp_c.npy'), dtype=torch.float)
    out_c = torch.tensor(np.load(f'{output_folder}/{dataset}/out_c.npy'), dtype=torch.float)
    return inp, out, inp_c, out_c

def init_impute(inp_c, out_c, inp_m, out_m, strategy = 'zero'):
    if strategy == 'zero':
        inp_r, out_r = torch.zeros(inp_c.shape), torch.zeros(out_c.shape)
    elif strategy == 'random':
        inp_r, out_r = torch.rand(inp_c.shape), torch.rand(out_c.shape)
    else:
        raise NotImplementedError()
    inp_r, out_r = inp_r.double(), out_r.double()
    inp_c[inp_m], out_c[out_m] = inp_r[inp_m], out_r[out_m]
    return inp_c, out_c
 
if __name__ == '__main__':
    from src.parser import *
    inp, out, inp_c, out_c = load_data(args.dataset)
    inp_m, out_m = torch.isnan(inp_c).float(), torch.isnan(out_c).float()
    # inp_c, out_c = init_impute(inp_c, out_c, inp_m, out_m, strategy = 'zero')

    dataloader = DataLoader(list(zip(inp, inp_c, out_c, (1-inp_m))), batch_size=128, shuffle=False)
    
    trainer = GAINTrainer(inp_c.shape[1], out_c.shape[1], {'min': torch.zeros(inp_c.shape[1]), 'max': torch.ones(inp_c.shape[1])}, args)
    print(trainer.train_model(dataloader, dataloader))
