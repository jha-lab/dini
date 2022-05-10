from src.utils import *
from src.folderconstants import *
from src.models import *
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from fancyimpute import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from corrupt import process as corrupt_process
from baseline import load_data as load_data_all, init_impute as init_impute_all
from gmm import correct_subset, opt as gmm_opt
from sklearn.mixture import GaussianMixture
from grape import load_data as load_data_sep, init_impute as init_impute_sep, train_gnn_mdi
from gain import GAINGenerator, GAINDiscriminator, GAINTrainer
from dini import load_model, save_model, backprop, opt

import sys
sys.path.append('./GRAPE/')
import pickle
import argparse
import pandas as pd
from uci.uci_data import get_data
from mc.mc_subparser import add_mc_subparser
from uci.uci_subparser import add_uci_subparser
from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge

from matplotlib import pyplot as plt

import warnings
from functools import partialmethod
warnings.filterwarnings('ignore')
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINI')
    parser.add_argument('--dataset', 
                        metavar='-d', 
                        type=str, 
                        required=False,
                        default='MSDS',
                        help="dataset from ['MSDS', 'concrete', 'linear']")
    parser.add_argument('--strategy', 
                        metavar='-s', 
                        type=str, 
                        required=False,
                        default='MCAR',
                        help="corruption strategy from ['MCAR', 'MAR', 'MNAR']")
    parser.add_argument('--fraction',
                        metavar='-f',
                        type=float,
                        required=False,
                        default=0.1,
                        help="fraction of data to corrupt; should be less than 1")
    args = parser.parse_args()

    # Corrupt the given dataset
    corrupt_process(args.dataset, args.strategy, args.fraction)

    data, data_c = load_data_all(args.dataset)
    data_m = np.isnan(data_c)
    
    data_c_imputed = init_impute_all(data_c, data_m, strategy = 'zero')

    print('Starting MSE:\t', mse(data[data_m], data_c_imputed[data_m]))
    print('Starting MAE:\t', mae(data[data_m], data_c_imputed[data_m]))

    results = {'corrupted': [mse(data[data_m], data_c_imputed[data_m]), mae(data[data_m], data_c_imputed[data_m])]}

    # Run baselines
    baseline_models = ['mean', 'median', 'knn', 'svd', 'mice', 'spectral', 'matrix']
    for model in baseline_models:
        if model == 'mean':
            data_new = SimpleFill(fill_method='mean').fit_transform(data_c)
        elif model == 'median':
            data_new = SimpleFill(fill_method='median').fit_transform(data_c)
        elif model == 'knn':
            k = [3,10,50][0]
            data_new = KNN(k=k, verbose=False).fit_transform(data_c)
        elif model == 'svd':
            rank = [np.ceil((data_c.shape[1]-1)/10),np.ceil((data_c.shape[1]-1)/5), data_c.shape[1]-1][0]
            data_new = IterativeSVD(rank=int(rank), verbose=False).fit_transform(data_c)
        elif model == 'mice':
            max_iter = [3,10,50][0]
            data_new = IterativeImputer(max_iter=max_iter).fit_transform(data_c)
        elif model == 'spectral':
            sparsity = [0.5,None,3][0]
            data_new = SoftImpute(shrinkage_value=sparsity, verbose=False).fit_transform(data_c)
        elif model == 'matrix':
            data_new = MatrixFactorization(verbose=False).fit_transform(data_c)
        else:
            raise NotImplementedError()

        print(f'{model.upper()} MSE:\t', mse(data[data_m], data_new[data_m]))
        print(f'{model.upper()} MAE:\t', mae(data[data_m], data_new[data_m]))

        results[model] = [mse(data[data_m], data_new[data_m]), mae(data[data_m], data_new[data_m])]

    # Run GMM
    subset = correct_subset(data_c_imputed, data_m)
    gm = GaussianMixture(n_components=50, random_state=0).fit(subset)

    data_new = gmm_opt(gm, data_c, data_m)
    print(f'GMM MSE:\t', mse(data[data_m], data_new[data_m]))
    print(f'GMM MAE:\t', mae(data[data_m], data_new[data_m]))

    results['gmm'] = [mse(data[data_m], data_new[data_m]), mae(data[data_m], data_new[data_m])]

    # Run GAIN
    inp, out, inp_c, out_c = load_data_sep(args.dataset)
    inp_m, out_m = torch.isnan(inp_c), torch.isnan(out_c)
    inp_c_imputed, out_c_imputed = init_impute_sep(inp_c, out_c, inp_m, out_m, strategy = 'zero')
    inp_m, out_m = inp_c.float(), out_c.float()

    dataloader = DataLoader(list(zip(inp, inp_c, out_c, (1-inp_m))), batch_size=128, shuffle=False)
    
    trainer = GAINTrainer(inp_c.shape[1], out_c.shape[1], {'min': torch.zeros(inp_c.shape[1]), 'max': torch.ones(inp_c.shape[1])}, args)
    perf_dict, imputed_data = trainer.train_model(dataloader, dataloader)
    data_new = torch.cat((imputed_data, out_c_imputed), dim=1)
    print(f'GAIN MSE:\t', mse(data[data_m], data_new[data_m]))
    print(f'GAIN MAE:\t', mae(data[data_m], data_new[data_m]))

    results['gain'] = [mse(data[data_m], data_new[data_m]), mae(data[data_m], data_new[data_m])]

    # Run GRAPE
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 1 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='mean',)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7) # 1 - edge dropout rate
    parser.add_argument('--auto_known', action='store_true', default=False)
    parser.add_argument('--loss_mode', type=int, default = 0) # 0: loss on all train edge, 1: loss only on unknown train edge
    parser.add_argument('--valid', type=float, default=0.) # valid-set ratio
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='log_dir')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_prediction', action='store_true', default=False)
    parser.add_argument('--transfer_dir', type=str, default=None)
    parser.add_argument('--transfer_extra', type=str, default='')
    parser.add_argument('--mode', type=str, default='train') # debug
    parser.add_argument('--dataset', type=str, default=args.dataset)
    parser.add_argument('--strategy', type=str, default=args.strategy)
    parser.add_argument('--fraction', type=str, default=args.fraction)
    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    add_mc_subparser(subparsers)
    grape_args = parser.parse_args()
    grape_args.dataset = args.dataset

    df_inp = pd.DataFrame(inp)
    df_out = pd.DataFrame(out)

    data_forward = get_data(df_inp, 
        df_out, 
        0, # node mode
        0.7, # train edge
        0, # split sample (percentage of model to corrupt)
        'random', # split by in ['y', 'random']
        0.9, # train_y prob.
        0)

    data_backward = get_data(df_out, 
        df_inp, 
        0, # node mode
        0.7, # train edge
        0.1, # split sample (percentage of model to corrupt)
        'random', # split by in ['y', 'random']
        0.9, # train_y prob.
        0)

    log_path = './temp/{}/'.format(grape_args.log_dir)
    os.makedirs(log_path, exist_ok=True)

    # print('Training forward model...')
    pred_train1, pred_test1, train_labels1, test_labels1 = train_gnn_mdi(data_forward, grape_args, log_path, torch.device('cpu'), verbose=False)

    # print('Training backward model...')
    grape_args.epochs = 1
    pred_train2, pred_test2, train_labels2, test_labels2 = train_gnn_mdi(data_backward, grape_args, log_path, torch.device('cpu'), verbose=False)

    pred = np.concatenate([pred_train1, pred_test1, pred_train2, pred_test2])
    labels = np.concatenate([train_labels1, test_labels1, train_labels2, test_labels2])

    print('GRAPE MSE:\t', mse(pred, labels))
    print('GRAPE MAE:\t', mae(pred, labels))

    results['grape'] = [mse(data[data_m], data_new[data_m]), mae(data[data_m], data_new[data_m])]

    # Run DINI
    num_epochs = 80
    lf = nn.MSELoss(reduction = 'mean')

    inp, out, inp_c, out_c = load_data_sep(args.dataset)
    inp_m, out_m = torch.isnan(inp_c), torch.isnan(out_c)
    inp_m2, out_m2 = torch.logical_not(inp_m), torch.logical_not(out_m)
    inp_c, out_c = init_impute_sep(inp_c, out_c, inp_m, out_m, strategy = 'zero')
    inp_c, out_c = inp_c.double(), out_c.double()
    model, optimizer, epoch, accuracy_list = load_model('FCN2', inp, out, args.dataset, True, False)
    data_c = torch.cat([inp_c, out_c], dim=1)
    data_m = torch.cat([inp_m, out_m], dim=1)
    data = torch.cat([inp, out], dim=1)

    early_stop_patience, curr_patience, old_loss = 3, 0, np.inf
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        dataloader = DataLoader(list(zip(inp_c, out_c, inp_m, out_m)), batch_size=1, shuffle=False)

        # Tune Model
        unfreeze_model(model)
        loss = backprop(e, model, optimizer, dataloader)
        accuracy_list.append(loss)
        save_model(model, optimizer, e, accuracy_list, args.dataset, 'FCN2')

        # Tune Data
        freeze_model(model)
        inp_c, out_c = opt(model, dataloader)
        data_c = torch.cat([inp_c, out_c], dim=1)
        tqdm.write(f'Epoch {e},\tLoss = {loss},\tMSE = {lf(data[data_m], data_c[data_m]).item()},\tMAE = {mae(data[data_m].detach().numpy(), data_c[data_m].detach().numpy())}')  

        if lf(data[data_m], data_c[data_m]).item() >= old_loss: curr_patience += 1
        if curr_patience > early_stop_patience: break
        old_loss = lf(data[data_m], data_c[data_m]).item()

    print('DINI MSE:\t', lf(data[data_m], data_c[data_m]).item())
    print('DINI MSE\t', mae(data[data_m].detach().numpy(), data_c[data_m].detach().numpy()))

    results['dini'] = [lf(data[data_m], data_c[data_m]).item(), mae(data[data_m].detach().numpy(), data_c[data_m].detach().numpy())]

    os.makedirs('./results', exist_ok=True)
    df = pd.DataFrame(results)
    df.index = ['mse', 'mae']

    fraction_str = str(args.fraction).split(".")[-1]
    df.to_csv(f'./results/{args.dataset.lower()}_{args.strategy.lower()}_p{fraction_str}.csv')

    fig, ax = plt.subplots(figsize=(18, 4.8))
    x = np.arange(len(results))
    width = 0.4

    # ax.bar(x, [results[key][0] for key in results.keys()], color='#4292C6', label='MSE')
    ax.bar(x - width*0.5, [results[key][0] for key in results.keys()], width, color='#F1C40F', label='MSE')
    ax.bar(x + width*0.5, [results[key][1] for key in results.keys()], width, color='#E67E22', label='MAE')
    ax.set_ylabel('Error')
    ax.set_xticks(x)
    ax.set_xticklabels([key.upper() for key in results.keys()])
    ax.legend(loc='upper right')
    plt.title(f'Dataset: {args.dataset.upper()} | Corruption strategy: {args.strategy.upper()} | Fraction: {args.fraction}')
    plt.grid(axis='y', linestyle='--')
    plt.savefig(f'./results/{args.dataset.lower()}_{args.strategy.lower()}_p{fraction_str}.pdf', bbox_inches='tight')

