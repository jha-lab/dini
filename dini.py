from src.models import *
from src.utils import *
from src.folderconstants import *
from src.adahessian import Adahessian
import torch
import numpy as np
from torch.utils.data import DataLoader
from fancyimpute import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tqdm import tqdm
from copy import deepcopy

torch.set_printoptions(sci_mode=True)

def sliding_windows(data, seq_length):
    x = []
    data_np = data.numpy()

    for i in range(0, len(data_np), seq_length):
        _x = data_np[i:(i+seq_length), :]
        x.append(_x)

    return torch.Tensor(np.array(x))

def load_data(dataset):
    inp = torch.tensor(np.load(f'{output_folder}/{dataset}/inp.npy')).float()
    out = torch.tensor(np.load(f'{output_folder}/{dataset}/out.npy')).float()
    inp_c = torch.tensor(np.load(f'{output_folder}/{dataset}/inp_c.npy')).float()
    out_c = torch.tensor(np.load(f'{output_folder}/{dataset}/out_c.npy')).float()
    return inp, out, inp_c, out_c

def init_impute(inp_c, out_c, inp_m, out_m, strategy = 'zero'):
    if strategy == 'zero':
        inp_r, out_r = torch.zeros(inp_c.shape), torch.zeros(out_c.shape)
    elif strategy == 'random':
        inp_r, out_r = torch.rand(inp_c.shape), torch.rand(out_c.shape)
    elif strategy == 'mean':
        inp_r, out_r = torch.Tensor(SimpleFill().fit_transform(inp_c)), torch.Tensor(SimpleFill().fit_transform(out_c))
    else:
        raise NotImplementedError()
    inp_r, out_r = inp_r, out_r
    inp_c[inp_m], out_c[out_m] = inp_r[inp_m], out_r[out_m]
    return inp_c, out_c

def load_model(modelname, inp, out, dataset, retrain, test):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(inp.shape[1], out.shape[1], 512)
    optimizer = torch.optim.Adam(model.parameters() , lr=0.0001, weight_decay=1e-3)
    fname = f'{checkpoints_folder}/{dataset}/{modelname}.ckpt'
    if os.path.exists(fname) and (not retrain or test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1; accuracy_list = []
    return model, optimizer, epoch, accuracy_list

def save_model(model, optimizer, epoch, accuracy_list, dataset, modelname):
    folder = f'{checkpoints_folder}/{dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}{modelname}.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def backprop(epoch, model, optimizer, dataloader, use_ce=False):
    lf = lambda x, y: nn.MSELoss(reduction = 'mean')(x, y) + nn.L1Loss(reduction = 'mean')(x, y)
    lfo = nn.CrossEntropyLoss(reduction = 'mean')
    ls = []
    for inp, out, inp_m, out_m in tqdm(dataloader, leave=False, ncols=80):
        pred_i, pred_o = model(inp.float(), out.float())
        loss = lf(pred_i, inp) + (lfo(pred_o, out) if use_ce else lf(pred_o, out))
        ls.append(loss.item())   
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return np.mean(ls)

def opt(model, dataloader, use_ce=False, use_second_order=False):
    lf = lambda x, y: nn.MSELoss(reduction = 'mean')(x, y) + nn.L1Loss(reduction = 'mean')(x, y)
    lfo = nn.CrossEntropyLoss(reduction = 'mean')
    ls = []; new_inp, new_out = [], []
    for inp, out, inp_m, out_m in tqdm(dataloader, leave=False, ncols=80):
        # update input
        inp.requires_grad = True; out.requires_grad = True
        optimizer = torch.optim.Adam([inp, out] , lr=0.0005) if not use_second_order else Adahessian([inp, out], lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        iteration = 0; equal = 0; z_old = 100
        inp_orig, out_orig = deepcopy(inp.detach().data), deepcopy(out.detach().data)
        while iteration < 800:
            inp_old = deepcopy(inp.data); out_old = deepcopy(out.data)
            pred_i, pred_o = model(inp, out)
            z =  lf(pred_i, inp) + (lfo(pred_o, out) if use_ce else lf(pred_o, out))
            optimizer.zero_grad(); z.backward(create_graph=True); optimizer.step(); scheduler.step()
            inp.data, out.data = scale(inp.data), scale(out.data)
            inp.data, out.data = mask(inp.data.detach(), inp_m, inp_orig), mask(out.data.detach(), out_m, out_orig)
            equal = equal + 1 if torch.all(torch.abs(inp_old - inp) < 0.01) and torch.all(torch.abs(out_old - out) < 0.01) else 0
            if equal > 30: break
            iteration += 1; z_old = z.item()
        ls.append(z.item())
        inp.requires_grad = False; out.requires_grad = False
        new_inp.append(inp); new_out.append(out)
    return torch.cat(new_inp), torch.cat(new_out)

def forward_opt(model, dataloader):
    new_inp, new_out = [], []
    for inp, out, inp_m, out_m in tqdm(dataloader, leave=False, ncols=80):
        iteration = 0; equal = 0
        eps = 1e-6
        inp_orig, out_orig = deepcopy(inp.detach().data), deepcopy(out.detach().data)
        while iteration < 1800:
            inp_old = deepcopy(inp.data); out_old = deepcopy(out.data)
            inp, out = model(inp, out)
            inp.data, out.data = mask(inp.data.detach(), inp_m, inp_orig), mask(out.data.detach(), out_m, out_orig)
            equal = equal + 1 if torch.all(torch.abs(inp_old - inp) < eps) and torch.all(torch.abs(out_old - out) < eps) else 0
            if equal > 30: break
            iteration += 1
        new_inp.append(inp); new_out.append(out)
    return torch.cat(new_inp), torch.cat(new_out)

if __name__ == '__main__':
    from src.parser import *
    num_epochs = 100 if not args.test else 0
    lf = nn.MSELoss(reduction = 'mean')

    inp, out, inp_c, out_c = load_data(args.dataset)
    inp_m, out_m = torch.isnan(inp_c), torch.isnan(out_c)
    inp_m2, out_m2 = torch.logical_not(inp_m), torch.logical_not(out_m)
    inp_c, out_c = init_impute(inp_c, out_c, inp_m, out_m, strategy = 'zero')
    model, optimizer, epoch, accuracy_list = load_model(args.model, inp, out, args.dataset, args.retrain, args.test)
    data_c = torch.cat([inp_c, out_c], dim=1)
    data_m = torch.cat([inp_m, out_m], dim=1)
    data = torch.cat([inp, out], dim=1)

    print('Starting MSE', lf(data[data_m], data_c[data_m]).item()) 

    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        if model.name.startswith('FCN'):
            dataloader = DataLoader(list(zip(inp_c, out_c, inp_m, out_m)), batch_size=1, shuffle=False)
        else:
            dataloader = DataLoader(list(zip(sliding_windows(inp_c, 5), sliding_windows(out_c, 5), sliding_windows(inp_m, 5), sliding_windows(out_m, 5))), batch_size=1, shuffle=False)

        # Tune Model
        unfreeze_model(model)
        loss = backprop(e, model, optimizer, dataloader)
        accuracy_list.append(loss)
        save_model(model, optimizer, e, accuracy_list, args.dataset, args.model)

        # Tune Data
        freeze_model(model)
        inp_c, out_c = opt(model, dataloader)
        
        if not model.name.startswith('FCN'):
            inp_c = inp_c.view(-1, inp_c.shape[-1])
            out_c = out_c.view(-1, out_c.shape[-1])

        data_c = torch.cat([inp_c, out_c], dim=1)
        tqdm.write(f'Epoch {e},\tLoss = {loss},\tMSE = {lf(data[data_m], data_c[data_m]).item()},\tMAE = {mae(data[data_m].detach().numpy(), data_c[data_m].detach().numpy())}')  
    