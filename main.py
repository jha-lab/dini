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
    optimizer = torch.optim.Adam(model.parameters() , lr=1e-4, weight_decay=1e-5)
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
        pred = model(inp)
        loss = lf(pred, out)
        ls.append(loss.item())   
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return np.mean(ls)

def opt(model, dataloader):
    lf = nn.MSELoss(reduction = 'mean')
    ls = []; new_inp = []
    for init, out in tqdm(dataloader, leave=False, ncols=80):
        # update input
        init.requires_grad = True
        optimizer = torch.optim.AdamW([init] , lr=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        iteration = 0; equal = 0; z_old = 100
        while iteration < 600:
            init_old = deepcopy(init.data)
            pred = model(init)
            z = lf(pred, out)
            optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
            init.data = scale(init.data)
            equal = equal + 1 if torch.all(init_old - init < 0.01) else 0
            if equal > 30: break
            iteration += 1; z_old = z.item()
        ls.append(z.item())
        init.requires_grad = False
        new_inp.append(init)
    return torch.cat(new_inp), np.mean(ls)
 
if __name__ == '__main__':
    inp, out, inp_c, out_c = load_data(args.dataset)
    inp_c, out_c = init_impute(inp_c, out_c, strategy = 'zero')
    model, optimizer, epoch, accuracy_list = load_model(args.model, inp, out)

    num_epochs = 5 if not args.test else 0
    lf = nn.MSELoss(reduction = 'mean')
    
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        dataloader = DataLoader(list(zip(inp_c, out_c)), batch_size=512, shuffle=True)

        # Tune Model
        unfreeze_model(model)
        loss = backprop(e, model, optimizer, dataloader)
        accuracy_list.append(loss)
        tqdm.write(f'Epoch {e},\tLoss = {loss}')
        save_model(model, optimizer, e, accuracy_list)

        # Tune Data
        freeze_model(model)
        inp_c, dev = opt(model, dataloader)
        tqdm.write(f'Epoch {e},\tMSE = {dev}')      
        