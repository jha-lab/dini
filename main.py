from src.models import *
from src.parser import *
from src.utils import *
from src.folderconstants import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

def load_data(dataset):
    inp = torch.tensor(np.load(f'{output_folder}/{dataset}/inp.npy'))
    out = torch.tensor(np.load(f'{output_folder}/{dataset}/out.npy'))
    inp_c = torch.tensor(np.load(f'{output_folder}/{dataset}/inp_c.npy'))
    out_c = torch.tensor(np.load(f'{output_folder}/{dataset}/out_c.npy'))
    return inp, out, inp_c, out_c

def init_impute(inp_c, out_c, strategy = 'zero'):
    if strategy == 'zero':
        return torch.nan_to_num(inp_c, nan=0), torch.nan_to_num(out_c, nan=0)
    else:
        raise NotImplementedError()

def load_model(modelname, inp, out):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(inp.shape[1], out.shape[1], 32).double()
    optimizer = torch.optim.Adam(model.parameters() , lr=1e-3, weight_decay=1e-5)
    fname = f'{checkpoints_folder}/{args.dataset}/{args.model}.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
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

def save_model(model, optimizer, epoch, accuracy_list):
    folder = f'{checkpoints_folder}/{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}{args.model}.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def backprop(epoch, model, optimizer, dataloader):
    lf = nn.MSELoss(reduction = 'mean')
    ls = []
    for inp, out in tqdm(dataloader, leave=False, ncols=80):
        pred_i, pred_o = model(inp, out)
        loss = lf(pred_o, out) + lf(pred_i, inp)
        ls.append(loss.item())   
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return np.mean(ls)

def opt(model, dataloader):
    lf = nn.MSELoss(reduction = 'mean')
    ls = []; new_inp, new_out = [], []
    for inp, out in tqdm(dataloader, leave=False, ncols=80):
        # update input
        inp.requires_grad = True; out.requires_grad = True
        optimizer = torch.optim.Adam([inp, out] , lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        iteration = 0; equal = 0; z_old = 100
        while iteration < 800:
            init_old = deepcopy(torch.cat([inp,out]).data)
            pred_i, pred_o = model(inp, out)
            z = lf(pred_o, out) + lf(pred_i, inp)
            optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
            inp.data, out.data = scale(inp.data), scale(out.data)
            equal = equal + 1 if torch.all(init_old - torch.cat([inp,out]) < 0.01) else 0
            if equal > 30: break
            iteration += 1; z_old = z.item()
        ls.append(z.item())
        inp.requires_grad = False; out.requires_grad = False
        new_inp.append(inp); new_out.append(out)
    return torch.cat(new_inp), torch.cat(new_out), np.mean(ls)
 
if __name__ == '__main__':
    inp, out, inp_c, out_c = load_data(args.dataset)
    inp_c, out_c = init_impute(inp_c, out_c, strategy = 'zero')
    model, optimizer, epoch, accuracy_list = load_model(args.model, inp, out)

    num_epochs = 20 if not args.test else 0
    lf = nn.MSELoss(reduction = 'mean')
    
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        dataloader = DataLoader(list(zip(inp_c, out_c)), batch_size=512, shuffle=True)

        # Tune Model
        unfreeze_model(model)
        loss = backprop(e, model, optimizer, dataloader)
        accuracy_list.append(loss)
        save_model(model, optimizer, e, accuracy_list)

        # Tune Data
        freeze_model(model)
        inp_c, out_c, loss = opt(model, dataloader)
        dev = lf(inp_c, inp) + lf(out_c, out)
        tqdm.write(f'Epoch {e},\tLoss = {loss},\tMSE = {dev}')      
        