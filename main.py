from src.utils import *
from src.folderconstants import *
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

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINI')
    parser.add_argument('--dataset', 
                        metavar='-d', 
                        type=str, 
                        required=True,
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
    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    add_mc_subparser(subparsers)
    grape_args = parser.parse_args()
    print(grape_args)

    inp, out, inp_c, out_c = load_data_sep(grape_args.dataset)
    inp_m, out_m = torch.isnan(inp_c).float(), torch.isnan(out_c).float()
    # inp_c, out_c = init_impute(inp_c, out_c, inp_m, out_m, strategy = 'zero')

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
    pred_train1, pred_test1, train_labels1, test_labels1 = train_gnn_mdi(data_forward, grape_args, log_path, torch.device('cpu'))

    # print('Training backward model...')
    grape_args.epochs = 1
    pred_train2, pred_test2, train_labels2, test_labels2 = train_gnn_mdi(data_backward, grape_args, log_path, torch.device('cpu'))

    pred = np.concatenate([pred_train1, pred_test1, pred_train2, pred_test2])
    labels = np.concatenate([train_labels1, test_labels1, train_labels2, test_labels2])

    print('GRAPE MSE: ', mse(pred, labels))
    print('GRAPE MAE: ', mae(pred, labels))
