from src.utils import *
from src.folderconstants import *
from src.models import *
import argparse
import torch
import shutil
import json
import numpy as np
from torch.utils.data import DataLoader
from fancyimpute import *
from collections import Counter
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from corrupt import process as corrupt_process, MCAR, MAR, MNAR, MPAR, MSAR, normalize
from baseline import load_data as load_data_all, init_impute as init_impute_all
from gmm import correct_subset, opt as gmm_opt
from sklearn.mixture import GaussianMixture
from impute import impute
from grape import load_data as load_data_sep, init_impute as init_impute_sep, train_gnn_mdi
from gain import GAINGenerator, GAINDiscriminator, GAINTrainer
from dini import load_model, save_model, backprop, opt
from sklearn.linear_model import LinearRegression, SGDRegressor
from torch.utils.data.dataloader import default_collate

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
from matplotlib.ticker import MaxNLocator
import multiprocessing

import warnings
from functools import partialmethod
warnings.filterwarnings('ignore')
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


PLOT_RESULTS = False
VERBOSE = False
UNIFORM_SAMPLING = True


def train_fcn(inp_imp, out_imp, inp_train, out_train, inp_test, out_test, num_epochs):
    model, optimizer, epoch, accuracy_list = load_model('FCN', inp_imp, out_imp, dataset, True, False)

    # Train model on imputed data
    early_stop_patience, curr_patience, old_loss = 3, 0, np.inf
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        dataloader = DataLoader(list(zip(inp_imp, out_imp)), batch_size=1, shuffle=False)

        # Tune Model
        ls = []
        for inp, out in tqdm(dataloader, leave=False, ncols=80):
            pred_i, pred_o = model(inp, out)
            loss = lfo(pred_o, out)
            ls.append(loss.item())   
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if VERBOSE: tqdm.write(f'Epoch {e},\tLoss = {np.mean(ls)}')

        if np.mean(ls) >= old_loss: curr_patience += 1
        if curr_patience > early_stop_patience: break
        old_loss = np.mean(ls)

    # Get accuracy on train set
    y_pred, y_true = [], []
    train_dataloader = DataLoader(list(zip(inp_train, out_train)), batch_size=1, shuffle=False)
    for inp, out in tqdm(train_dataloader, leave=False, ncols=80):
        pred_i, pred_o = model(inp, torch.zeros_like(out))
        if out.shape[1] > 1:
            y_pred.append(np.argmax(pred_o.detach().numpy()))
            y_true.append(np.argmax(out))
        else:
            y_pred.append(int(np.around(pred_o.detach().numpy())))
            y_true.append(int(out))

    train_accuracy = accuracy_score(y_true, y_pred)

    # Get accuracy on test set
    y_pred, y_true = [], []
    test_dataloader = DataLoader(list(zip(inp_test, out_test)), batch_size=1, shuffle=False)
    for inp, out in tqdm(test_dataloader, leave=False, ncols=80):
        pred_i, pred_o = model(inp.float(), torch.zeros_like(out).float())
        if out.shape[1] > 1:
            y_pred.append(np.argmax(pred_o.detach().numpy()))
            y_true.append(np.argmax(out))
        else:
            y_pred.append(int(np.around(pred_o.detach().numpy())))
            y_true.append(int(out))

    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, average = 'binary')
    test_recall = recall_score(y_true, y_pred, average = 'binary')
    test_f1_score = f1_score(y_true, y_pred, average = 'binary')

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1))

    return train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, cm


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='DINI')
    parser.add_argument('--dataset', 
                        metavar='-d', 
                        type=str, 
                        required=False,
                        default='cps_wdt',
                        help="dataset from ['gas', 'swat', 'coviddeep', 'covid_cxr']")
    parser.add_argument('--strategy', 
                        metavar='-s', 
                        type=str, 
                        required=False,
                        default='MPAR',
                        help="corruption strategy from ['MCAR', 'MAR', 'MNAR', 'MPAR', 'MSAR']")
    parser.add_argument('--fraction',
                        metavar='-f',
                        type=float,
                        required=False,
                        default=1,
                        help="fraction of data to corrupt; should be less than 1")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    # Create train, corrupt, test splits
    dataset = args.dataset
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    data_file = f'{data_folder}/{dataset}/data.csv'
    df = pd.read_csv(data_file, index_col=0 if dataset != 'covid_cxr' else None)
    assert not np.any(np.isnan(df.values))
    df = normalize(df)

    if UNIFORM_SAMPLING:
        if dataset == 'gas':
            df = pd.concat([df.loc[df['Gas'] == i].sample(500, random_state=0) for i in [0, 1]])
        elif dataset == 'swat':
            df = pd.concat([df.loc[df['Normal/Attack'] == i].sample(100, random_state=0) for i in [0, 1]])
        elif dataset == 'coviddeep':
            df = pd.concat([df.loc[df['4163'] == i].sample(500, random_state=0) for i in [0, 1]])
        elif dataset == 'covid_cxr':
            df = pd.concat([df.loc[df['512'] == i].sample(500, random_state=0, replace=True) for i in [0, 1]])
        df = df.sample(frac=1, random_state=0) # remove random_state=0 for CI bounds
    else:
        df = df.sample(frac=0.01, random_state=0) # randomize dataset

    df_size = df.values.shape[0]
    print(f'Sampled dataset size: {df_size}')

    # 40-40-20 split of the dataset
    df_train = df.iloc[:int(0.4*df_size), :]
    df_val = df.iloc[int(0.4*df_size):int(0.8*df_size), :]
    df_test = df.iloc[int(0.8*df_size):, :]

    corruption, fraction = args.strategy, args.fraction
    if corruption == 'MCAR':
        corrupt_df = MCAR(df_val, fraction)
    elif corruption == 'MAR':
        corrupt_df = MAR(df_val, fraction)
    elif corruption == 'MNAR':
        corrupt_df = MNAR(df_val, fraction)
    elif corruption == 'MPAR':
        corrupt_df = MPAR(df_val, fraction)
    elif corruption == 'MSAR':
        corrupt_df = MSAR(df_val, fraction)
    else:
        raise NotImplementedError()

    label_idx = -1
    inp_train, out_train = torch.tensor(df_train.values[:, :label_idx]).float(), torch.tensor(df_train.values[:, label_idx:]).float()
    inp_val, out_val = torch.tensor(df_val.values[:, :label_idx]).float(), torch.tensor(df_val.values[:, label_idx:]).float()
    inp_c, out_c = torch.tensor(corrupt_df.values[:, :label_idx]), torch.tensor(corrupt_df.values[:, label_idx:])
    inp_test, out_test = torch.tensor(df_test.values[:, :label_idx]), torch.tensor(df_test.values[:, label_idx:])

    lf = nn.MSELoss(reduction = 'mean')
    if label_idx == -1:
        lfo = nn.BCELoss(reduction = 'mean')
    else:
        lfo = nn.CrossEntropyLoss(reduction = 'mean')

    if PLOT_RESULTS and os.path.exists(f'./results/model/{dataset}/{corruption}/'): shutil.rmtree(f'./results/model/{dataset}/{corruption}/')

    # Train FCN on uncorrupted data
    train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, cm = train_fcn(torch.cat([inp_train, inp_val]), torch.cat([out_train, out_val]), inp_train, out_train, inp_test, out_test, 100)

    print(f'Uncorrupted Train Accuracy: {train_accuracy*100 : 0.2f}%')
    print(f'Uncorrupted Test Accuracy: {test_accuracy*100 : 0.2f}%')
    print(f'Uncorrupted Precision: {test_precision : 0.2f}')
    print(f'Uncorrupted Recall: {test_recall : 0.2f}')
    print(f'Uncorrupted F1 Score: {test_f1_score : 0.2f}')
    print(f'Uncorrupted Confusion Matrix:\n{cm}')

    results = {'uncorrupted': {'test_acc': test_accuracy*100, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1_score}}

    if PLOT_RESULTS:
        os.makedirs(f'./results/model/{dataset}/{corruption}/cms/', exist_ok=True)
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.imshow(cm)
        plt.savefig(f'./results/model/{dataset}/{corruption}/cms/unc.pdf', bbox_inches='tight')

    # Run DINI
    inp_dini, out_dini = torch.cat((inp_train, inp_c)), torch.cat((out_train, out_c))
    inp_m, out_m = torch.isnan(inp_dini), torch.isnan(out_dini)
    inp_m2, out_m2 = torch.logical_not(inp_m), torch.logical_not(out_m)
    inp_imp, out_imp = init_impute_sep(inp_dini, out_dini, inp_m, out_m, strategy = 'zero')
    inp_imp, out_imp = inp_imp.float(), out_imp.float()
    model, optimizer, epoch, accuracy_list = load_model('FCN2', inp_dini, out_dini, dataset, True, False)
    data_imp = torch.cat([inp_imp, out_imp], dim=1)
    data_m = torch.cat([inp_m, out_m], dim=1)
    data = torch.cat([torch.cat((inp_train, inp_val)), torch.cat((out_train, out_val))], dim=1)

    if PLOT_RESULTS:
        os.makedirs(f'./results/model/{dataset}/{corruption}/heatmaps/', exist_ok=True)
        plt.imshow(data)
        plt.savefig(f'./results/model/{dataset}/{corruption}/heatmaps/orig.pdf', bbox_inches='tight')
        # np.save(f'./results/model/{dataset}/{corruption}/heatmaps/orig.npy', data.numpy())
        # np.save(f'./results/model/{dataset}/{corruption}/heatmaps/corrupt.npy', torch.cat([inp_dini, out_dini], dim=1).numpy())

    num_epochs = 10

    ls = []
    early_stop_patience, curr_patience, old_loss = 5, 0, np.inf
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        dataloader = DataLoader(list(zip(inp_imp, out_imp, inp_m, out_m)), batch_size=1, shuffle=False)

        # Tune Model
        unfreeze_model(model)
        loss = backprop(e, model, optimizer, dataloader)
        accuracy_list.append(loss)
        save_model(model, optimizer, e, accuracy_list, dataset, f'FCN2_{args.strategy}')

        # Tune Data
        freeze_model(model)
        inp_imp, out_imp, _, _ = opt(model, dataloader)
        data_imp = torch.cat([inp_imp, out_imp], dim=1)
        if VERBOSE: tqdm.write(f'Epoch {e},\tLoss = {loss},\tMSE = {mse(data[data_m].detach().numpy(), data_imp[data_m].detach().numpy())},\tMAE = {mae(data[data_m].detach().numpy(), data_imp[data_m].detach().numpy())}')  

        ls.append(lf(data[data_m], data_imp[data_m]).item())
        if np.mean(ls[-10:]) >= old_loss: curr_patience += 1
        if curr_patience > early_stop_patience: break
        old_loss = np.mean(ls[-10:])

        # Save imputed data heatmap
        if e%10 == 0 and PLOT_RESULTS:
            plt.imshow(data_imp)
            plt.savefig(f'./results/model/{dataset}/{corruption}/heatmaps/dini_e{e}.pdf', bbox_inches='tight')
            # np.save(f'./results/model/{dataset}/{corruption}/heatmaps/dini_e{e}.npy', data_imp.numpy())

    if out_imp.shape[1] > 1: out_imp = torch.nn.functional.gumbel_softmax(out_imp, hard=True)
    data_imp[:, label_idx:] = torch.nn.functional.gumbel_softmax(data_imp[:, label_idx:], hard=True)
    unfreeze_model(model)

    if VERBOSE:
        print('DINI MSE:\t', mse(data[data_m].detach().numpy(), data_imp[data_m].detach().numpy()))
        print('DINI MAE\t', mae(data[data_m].detach().numpy(), data_imp[data_m].detach().numpy()))

    # Train FCN on DINI imputed data
    train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, cm = train_fcn(inp_imp, out_imp, inp_train, out_train, inp_test, out_test, 100)

    print(f'DINI Train Accuracy: {train_accuracy*100 : 0.2f}%')
    print(f'DINI Test Accuracy: {test_accuracy*100 : 0.2f}%')
    print(f'DINI Precision: {test_precision : 0.2f}')
    print(f'DINI Recall: {test_recall : 0.2f}')
    print(f'DINI F1 Score: {test_f1_score : 0.2f}')
    print(f'DINI Confusion Matrix:\n{cm}')

    results['dini'] = {'test_acc': test_accuracy*100, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1_score}

    if PLOT_RESULTS:
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.imshow(cm)
        plt.savefig(f'./results/model/{dataset}/{corruption}/cms/dini.pdf', bbox_inches='tight')

    # Run simple FCN on data imputed by baseline imputation methods
    baseline_models = ['median', 'knn', 'svd', 'mice', 'spectral', 'matrix', 'gain']
    for model in baseline_models:
        data_imp_base = torch.Tensor(impute(inp_c, out_c, model))
        data_imp_base_m = np.isnan(data_imp_base.numpy())
        data_imp_base = torch.Tensor(init_impute_all(data_imp_base.numpy(), data_imp_base_m, strategy = 'zero'))
        data_base = data_imp_base.double()
        inp_base, out_base = data_base[:, :label_idx], data_base[:, label_idx:]
        inp_base, out_base = torch.cat([inp_train, inp_base]).float(), torch.cat([out_train, out_base]).float()
        data_base = torch.cat([inp_base, out_base], dim=1)

        if PLOT_RESULTS:
            plt.imshow(data_base)
            os.makedirs(f'./results/model/{dataset}/{corruption}/heatmaps/', exist_ok=True)
            plt.savefig(f'./results/model/{dataset}/{corruption}/heatmaps/{model}.pdf', bbox_inches='tight')

        if VERBOSE:
            print(f'{model.upper()} MSE:\t', mse(data[data_m], data_base[data_m]))
            print(f'{model.upper()} MAE:\t', mae(data[data_m], data_base[data_m]))

        # Train FCN on baseline imputed data
        train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, cm = train_fcn(inp_base, out_base, inp_train, out_train, inp_test, out_test, 5)

        print(f'{model.upper()} Train Accuracy: {train_accuracy*100 : 0.2f}%')
        print(f'{model.upper()} Test Accuracy: {test_accuracy*100 : 0.2f}%')
        print(f'{model.upper()} Precision: {test_precision : 0.2f}')
        print(f'{model.upper()} Recall: {test_recall : 0.2f}')
        print(f'{model.upper()} F1 Score: {test_f1_score : 0.2f}')
        print(f'{model.upper()} Confusion Matrix:\n{cm}')

        results[model] = {'test_acc': test_accuracy*100, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1_score}

        if PLOT_RESULTS:
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.imshow(cm)
            plt.savefig(f'./results/model/{dataset}/{corruption}/cms/{model}.pdf', bbox_inches='tight')

    json.dump(results, open(f'./results/model/{dataset}/{corruption}/results.json', 'w+'))

    if PLOT_RESULTS:
        # Plot results
        fig, ax = plt.subplots(figsize=(18, 4.8))
        x = np.arange(len(results)-1)
        width = 0.4
        ax2 = ax.twinx()

        # ax.bar(x, [results[key][0] for key in results.keys()], color='#4292C6', label='MSE')
        ax.bar(x, [results[key]['test_acc'] for key in baseline_models + ['dini']], width, color='#F1C40F', label='Test Accuracy')
        ax2 .plot(x, [results[key]['f1'] for key in baseline_models + ['dini']], '.-', markersize=18, linewidth=4, color='#E67E22', label='F1 Score')
        ax.set_ylabel('Test Accuracy (%)')
        ax2.set_ylabel('F1 Score')
        ax.set_xticks(x)
        ax.set_xticklabels([model.upper() for model in baseline_models] + ['DINI'])
        plt.title(f'Dataset: {dataset.upper()} | Corruption strategy: {args.strategy.upper()}')
        plt.grid(axis='y', linestyle='--')
        plt.savefig(f'./results/model/{dataset}/{corruption}/barplot.pdf', bbox_inches='tight')


