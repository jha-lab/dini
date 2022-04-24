import numpy as np
from src.parser import *
from src.utils import *
from src.folderconstants import *
from sklearn.mixture import GaussianMixture

def load_data(dataset):
    inp = np.load(f'{output_folder}/{dataset}/inp.npy')
    out = np.load(f'{output_folder}/{dataset}/out.npy')
    inp_c = np.load(f'{output_folder}/{dataset}/inp_c.npy')
    out_c = np.load(f'{output_folder}/{dataset}/out_c.npy')
    data = np.concatenate([inp, out], axis=1)
    data_c = np.concatenate([inp_c, out_c], axis=1)
    return data, data_c

def init_impute(data_c, data_m, strategy = 'zero'):
    if strategy == 'zero':
        data_r = np.zeros(data_c.shape)
    elif strategy == 'random':
        data_r = np.random.random(data_c.shape)
    else:
        raise NotImplementedError()
    data_c[data_m] = data_r[data_m]
    return data_c

def correct_subset(data_c, data_m):
    subset = []
    for i in range(data_c.shape[0]):
        if not np.any(data_m[i]):
            subset.append(data_c[i])
    return np.array(subset)

def opt(model, dataloader):
    lf = nn.MSELoss(reduction = 'mean')
    ls = []; new_inp, new_out = [], []
    for inp, out, inp_m, out_m in tqdm(dataloader, leave=False, ncols=80):
        # update input
        inp.requires_grad = True; out.requires_grad = True
        optimizer = torch.optim.Adam([inp, out] , lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        iteration = 0; equal = 0; z_old = 100
        inp_orig, out_orig = deepcopy(inp.detach().data), deepcopy(out.detach().data)
        while iteration < 800:
            inp_old = deepcopy(inp.data); out_old = deepcopy(out.data)
            pred_i, pred_o = model(inp, out)
            z = lf(pred_o, out) + lf(pred_i, inp)
            optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
            inp.data, out.data = scale(inp.data), scale(out.data)
            inp.data, out.data = mask(inp.data.detach(), inp_m, inp_orig), mask(out.data.detach(), out_m, out_orig)
            equal = equal + 1 if torch.all(torch.abs(inp_old - inp) < 0.01) and torch.all(torch.abs(out_old - out) < 0.01) else 0
            if equal > 30: break
            iteration += 1; z_old = z.item()
        ls.append(z.item())
        inp.requires_grad = False; out.requires_grad = False
        new_inp.append(inp); new_out.append(out)
    return torch.cat(new_inp), torch.cat(new_out), np.mean(ls)
 
if __name__ == '__main__':
    data, data_c = load_data(args.dataset)
    data_m = np.isnan(data_c)
    data_c = init_impute(data_c, data_m, strategy = 'zero')
    subset = correct_subset(data_c, data_m)
    gm = GaussianMixture(n_components=20, random_state=0).fit(subset)
    print(gm.predict(data_c))
