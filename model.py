from src.utils import *
from src.folderconstants import *
from src.models import *
import argparse
import torch
import shutil
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


SAVE_RESULTS = True
UNIFORM_SAMPLING = True


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='DINI')
    parser.add_argument('--dataset', 
                        metavar='-d', 
                        type=str, 
                        required=False,
                        default='cps_wdt',
                        help="dataset from ['cps_wdt', 'gas', 'swat']")
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
    df = pd.read_csv(data_file, index_col=0)
    assert not np.any(np.isnan(df.values))
    df = normalize(df)

    if UNIFORM_SAMPLING:
        if dataset == 'cps_wdt':
            df = pd.concat([df.loc[df[class_name] == 1].sample(7) for class_name in ['Label_N', 'Label_DoS', 'Label_MITM', 'Label_S', 'Label_PF']])
        elif dataset == 'gas':
            df = pd.concat([df.loc[df['Gas'] == i].sample(500, random_state=0) for i in [0, 1]])
            df = df.sample(frac=1, random_state=0)
        elif dataset == 'swat':
            df = pd.concat([df.loc[df['Normal/Attack'] == i].sample(100, random_state=0) for i in [0, 1]])
            df = df.sample(frac=1, random_state=0)
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

    if dataset == 'cps_wdt':
        label_idx = -5
    elif dataset == 'gas':
        label_idx = -1
    elif dataset == 'swat':
        label_idx = -1

    inp_train, out_train = torch.tensor(df_train.values[:, :label_idx]), torch.tensor(df_train.values[:, label_idx:])
    inp_val, out_val = torch.tensor(df_val.values[:, :label_idx]), torch.tensor(df_val.values[:, label_idx:])
    inp_c, out_c = torch.tensor(corrupt_df.values[:, :label_idx]), torch.tensor(corrupt_df.values[:, label_idx:])
    inp_test, out_test = torch.tensor(df_test.values[:, :label_idx]), torch.tensor(df_test.values[:, label_idx:])

    num_epochs = 50
    lf = nn.MSELoss(reduction = 'mean')
    if label_idx == -1:
        lfo = nn.BCELoss(reduction = 'mean')
    else:
        lfo = nn.CrossEntropyLoss(reduction = 'mean')
    unc_model, optimizer, epoch, accuracy_list = load_model('FCN', inp_train, out_train, dataset, True, False)

    if SAVE_RESULTS and os.path.exists(f'./results/model/{dataset}/'): shutil.rmtree(f'./results/model/{dataset}')

    # Train model on uncorrupted data
    early_stop_patience, curr_patience, old_loss = 3, 0, np.inf
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        dataloader = DataLoader(list(zip(torch.cat([inp_train, inp_val]), torch.cat([out_train, out_val]))), batch_size=1, shuffle=False)

        # Tune Model
        ls = []
        for inp, out in tqdm(dataloader, leave=False, ncols=80):
            pred_i, pred_o = unc_model(inp, out)
            loss = lfo(pred_o, out)
            ls.append(loss.item())   
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        tqdm.write(f'Epoch {e},\tLoss = {np.mean(ls)}')

        if np.mean(ls) >= old_loss: curr_patience += 1
        if curr_patience > early_stop_patience: break
        old_loss = np.mean(ls)

    # Get accuracy on train set
    y_pred, y_true = [], []
    train_dataloader = DataLoader(list(zip(inp_train, out_train)), batch_size=1, shuffle=False)
    for inp, out in tqdm(train_dataloader, leave=False, ncols=80):
        pred_i, pred_o = unc_model(inp, torch.zeros_like(out))
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
        pred_i, pred_o = unc_model(inp, torch.zeros_like(out))
        if out.shape[1] > 1:
            y_pred.append(np.argmax(pred_o.detach().numpy()))
            y_true.append(np.argmax(out))
        else:
            y_pred.append(int(np.around(pred_o.detach().numpy())))
            y_true.append(int(out))

    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, average = 'micro')
    test_recall = recall_score(y_true, y_pred, average = 'micro')
    test_f1_score = f1_score(y_true, y_pred, average = 'micro')

    print(f'Uncorrupted Train Accuracy: {train_accuracy*100 : 0.2f}%')
    print(f'Uncorrupted Test Accuracy: {test_accuracy*100 : 0.2f}%')
    print(f'Uncorrupted Precision: {test_precision : 0.2f}')
    print(f'Uncorrupted Recall: {test_recall : 0.2f}')
    print(f'Uncorrupted F1 Score: {test_f1_score : 0.2f}')
    print(f'Uncorrupted Confusion Matrix:\n{confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1))}')

    results = {'uncorrupted': {'test_acc': test_accuracy*100, 'f1': test_f1_score}}

    if SAVE_RESULTS:
        os.makedirs(f'./results/model/{dataset}/cms/', exist_ok=True)
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.imshow(confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1)))
        plt.savefig(f'./results/model/{dataset}/cms/unc.pdf', bbox_inches='tight')

    # Run DINI
    inp_dini, out_dini = torch.cat((inp_train, inp_c)).double(), torch.cat((out_train, out_c)).double()
    inp_m, out_m = torch.isnan(inp_dini), torch.isnan(out_dini)
    inp_m2, out_m2 = torch.logical_not(inp_m), torch.logical_not(out_m)
    inp_imp, out_imp = init_impute_sep(inp_dini, out_dini, inp_m, out_m, strategy = 'zero')
    inp_imp, out_imp = inp_imp.double(), out_imp.double()
    model, optimizer, epoch, accuracy_list = load_model('FCN2', inp_dini, out_dini, dataset, True, False)
    data_imp = torch.cat([inp_imp, out_imp], dim=1)
    data_m = torch.cat([inp_m, out_m], dim=1)
    data = torch.cat([torch.cat((inp_train, inp_val)), torch.cat((out_train, out_val))], dim=1)

    if SAVE_RESULTS:
        os.makedirs(f'./results/model/{dataset}/heatmaps/', exist_ok=True)
        plt.imshow(data)
        plt.savefig(f'./results/model/{dataset}/heatmaps/orig.pdf', bbox_inches='tight')

    num_epochs = 300

    early_stop_patience, curr_patience, old_loss = 3, 0, np.inf
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        dataloader = DataLoader(list(zip(inp_imp, out_imp, inp_m, out_m)), batch_size=64, shuffle=False)

        # Tune Model
        unfreeze_model(model)
        loss = backprop(e, model, optimizer, dataloader)
        accuracy_list.append(loss)
        save_model(model, optimizer, e, accuracy_list, dataset, f'FCN2_{args.strategy}')

        # Tune Data
        freeze_model(model)
        inp_imp, out_imp = opt(model, dataloader)
        data_imp = torch.cat([inp_imp, out_imp], dim=1)
        tqdm.write(f'Epoch {e},\tLoss = {loss},\tMSE = {mse(data[data_m].detach().numpy(), data_imp[data_m].detach().numpy())},\tMAE = {mae(data[data_m].detach().numpy(), data_imp[data_m].detach().numpy())}')  

        if lf(data[data_m], data_imp[data_m]).item() >= old_loss: curr_patience += 1
        if curr_patience > early_stop_patience: break
        old_loss = lf(data[data_m], data_imp[data_m]).item()

        # Save imputed data heatmap
        if e%10 == 0 and SAVE_RESULTS:
            plt.imshow(data_imp)
            plt.savefig(f'./results/model/{dataset}/heatmaps/dini_e{e}.pdf', bbox_inches='tight')

    if out_imp.shape[1] > 1: out_imp = torch.nn.functional.gumbel_softmax(out_imp, hard=True)
    data_imp[:, label_idx:] = torch.nn.functional.gumbel_softmax(data_imp[:, label_idx:], hard=True)
    unfreeze_model(model)

    print('DINI MSE:\t', mse(data[data_m].detach().numpy(), data_imp[data_m].detach().numpy()))
    print('DINI MAE\t', mae(data[data_m].detach().numpy(), data_imp[data_m].detach().numpy()))

    num_epochs = 50
    train_model, optimizer, epoch, accuracy_list = load_model('FCN', inp_imp, out_imp, dataset, True, False)

    # Train model on imputed data
    early_stop_patience, curr_patience, old_loss = 3, 0, np.inf
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
        # Get Data
        dataloader = DataLoader(list(zip(inp_imp, out_imp, inp_m, out_m)), batch_size=1, shuffle=False)

        # Tune Model
        ls = []
        for inp, out, inp_m, out_m in tqdm(dataloader, leave=False, ncols=80):
            pred_i, pred_o = train_model(inp, out)
            loss = lfo(pred_o, out)
            ls.append(loss.item())   
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        tqdm.write(f'Epoch {e},\tLoss = {np.mean(ls)}')

        if np.mean(ls) >= old_loss: curr_patience += 1
        if curr_patience > early_stop_patience: break
        old_loss = np.mean(ls)

    # Get accuracy on train set
    y_pred, y_true = [], []
    train_dataloader = DataLoader(list(zip(inp_train, out_train)), batch_size=1, shuffle=False)
    for inp, out in tqdm(train_dataloader, leave=False, ncols=80):
        pred_i, pred_o = train_model(inp, torch.zeros_like(out))
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
        pred_i, pred_o = train_model(inp, torch.zeros_like(out))
        if out.shape[1] > 1:
            y_pred.append(np.argmax(pred_o.detach().numpy()))
            y_true.append(np.argmax(out))
        else:
            y_pred.append(int(np.around(pred_o.detach().numpy())))
            y_true.append(int(out))

    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, average = 'micro')
    test_recall = recall_score(y_true, y_pred, average = 'micro')
    test_f1_score = f1_score(y_true, y_pred, average = 'micro')

    print(f'DINI Train Accuracy: {train_accuracy*100 : 0.2f}%')
    print(f'DINI Test Accuracy: {test_accuracy*100 : 0.2f}%')
    print(f'DINI Precision: {test_precision : 0.2f}')
    print(f'DINI Recall: {test_recall : 0.2f}')
    print(f'DINI F1 Score: {test_f1_score : 0.2f}')
    print(f'DINI Confusion Matrix:\n{confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1))}')

    results['dini'] = {'test_acc': test_accuracy*100, 'f1': test_f1_score}

    if SAVE_RESULTS:
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.imshow(confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1)))
        plt.savefig(f'./results/model/{dataset}/cms/dini.pdf', bbox_inches='tight')

    # Run simple FCN on data imputed by baseline imputation methods
    baseline_models = ['median', 'knn', 'svd', 'mice', 'spectral', 'matrix', 'gmm', 'gain']
    for model in baseline_models:
        data_imp_base = torch.Tensor(impute(torch.cat([inp_train, inp_c]), torch.cat([out_train, out_c]), model))
        data_imp_base_m = np.isnan(data_imp_base.numpy())
        data_imp_base = torch.Tensor(init_impute_all(data_imp_base.numpy(), data_imp_base_m, strategy = 'zero'))
        data_base = data_imp_base.double()
        inp_base, out_base = data_base[:, :label_idx], data_base[:, label_idx:]

        if SAVE_RESULTS:
            plt.imshow(data_base)
            os.makedirs(f'./results/model/{dataset}/heatmaps/', exist_ok=True)
            plt.savefig(f'./results/model/{dataset}/heatmaps/{model}.pdf', bbox_inches='tight')

        print(f'{model.upper()} MSE:\t', mse(data[data_m], data_base[data_m]))
        print(f'{model.upper()} MAE:\t', mae(data[data_m], data_base[data_m]))

        baseline_model, optimizer, epoch, accuracy_list = load_model('FCN', inp_base, out_base, dataset, True, False)
        num_epochs = 50

        early_stop_patience, curr_patience, old_loss = 3, 0, np.inf
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
            # Get Data
            dataloader = DataLoader(list(zip(inp_base, out_base, inp_m, out_m)), batch_size=1, shuffle=False)

            # Tune Model
            ls = []
            for inp, out, inp_m, out_m in tqdm(dataloader, leave=False, ncols=80):
                pred_i, pred_o = baseline_model(inp, out)
                loss = lfo(pred_o, out)
                ls.append(loss.item())   
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            tqdm.write(f'Epoch {e},\tLoss = {np.mean(ls)}')

            if np.mean(ls) >= old_loss: curr_patience += 1
            if curr_patience > early_stop_patience: break
            old_loss = np.mean(ls)

       # Get accuracy on train set
        y_pred, y_true = [], []
        train_dataloader = DataLoader(list(zip(inp_train, out_train)), batch_size=1, shuffle=False)
        for inp, out in tqdm(train_dataloader, leave=False, ncols=80):
            pred_i, pred_o = baseline_model(inp, torch.zeros_like(out))
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
            pred_i, pred_o = baseline_model(inp, torch.zeros_like(out))
            if out.shape[1] > 1:
                y_pred.append(np.argmax(pred_o.detach().numpy()))
                y_true.append(np.argmax(out))
            else:
                y_pred.append(int(np.around(pred_o.detach().numpy())))
                y_true.append(int(out))

        test_accuracy = accuracy_score(y_true, y_pred)
        test_precision = precision_score(y_true, y_pred, average = 'micro')
        test_recall = recall_score(y_true, y_pred, average = 'micro')
        test_f1_score = f1_score(y_true, y_pred, average = 'micro')

        print(f'{model.upper()} Train Accuracy: {train_accuracy*100 : 0.2f}%')
        print(f'{model.upper()} Test Accuracy: {test_accuracy*100 : 0.2f}%')
        print(f'{model.upper()} Precision: {test_precision : 0.2f}')
        print(f'{model.upper()} Recall: {test_recall : 0.2f}')
        print(f'{model.upper()} F1 Score: {test_f1_score : 0.2f}')
        print(f'{model.upper()} Confusion Matrix:\n{confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1))}')

        results[model] = {'test_acc': test_accuracy*100, 'f1': test_f1_score}

        if SAVE_RESULTS:
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.imshow(confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1)))
            plt.savefig(f'./results/model/{dataset}/cms/{model}.pdf', bbox_inches='tight')

   #  # Train a model only on the uncorrupted dataset
   #  baseline_model, optimizer, epoch, accuracy_list = load_model('FCN', inp_train, out_train, dataset, True, False)
   #  num_epochs = 50

   #  early_stop_patience, curr_patience, old_loss = 3, 0, np.inf
   #  for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80):
   #      # Get Data
   #      dataloader = DataLoader(list(zip(inp_train, out_train)), batch_size=1, shuffle=False)

   #      # Tune Model
   #      ls = []
   #      for inp, out in tqdm(dataloader, leave=False, ncols=80):
   #          pred_i, pred_o = baseline_model(inp, out)
   #          loss = lfo(pred_o, out)
   #          ls.append(loss.item())   
   #          optimizer.zero_grad(); loss.backward(); optimizer.step()
        
   #      tqdm.write(f'Epoch {e},\tLoss = {np.mean(ls)}')

   #      if np.mean(ls) >= old_loss: curr_patience += 1
   #      if curr_patience > early_stop_patience: break
   #      old_loss = np.mean(ls)

   # # Get accuracy on train set
   #  y_pred, y_true = [], []
   #  train_dataloader = DataLoader(list(zip(inp_train, out_train)), batch_size=1, shuffle=False)
   #  for inp, out in tqdm(train_dataloader, leave=False, ncols=80):
   #      pred_i, pred_o = baseline_model(inp, torch.zeros_like(out))
   #      y_pred.append(np.argmax(pred_o.detach().numpy()))
   #      y_true.append(np.argmax(out))

   #  train_accuracy = accuracy_score(y_true, y_pred)

   #  # Get accuracy on test set
   #  y_pred, y_true = [], []
   #  test_dataloader = DataLoader(list(zip(inp_test, out_test)), batch_size=1, shuffle=False)
   #  for inp, out in tqdm(test_dataloader, leave=False, ncols=80):
   #      pred_i, pred_o = baseline_model(inp, torch.zeros_like(out))
   #      if out.shape[1] > 1:
   #          y_pred.append(np.argmax(pred_o.detach().numpy()))
   #          y_true.append(np.argmax(out))
   #      else:
   #          y_pred.append(int(np.around(pred_o.detach().numpy())))
   #          y_true.append(int(out))

   #  test_accuracy = accuracy_score(y_true, y_pred)
   #  test_precision = precision_score(y_true, y_pred, average = 'micro')
   #  test_recall = recall_score(y_true, y_pred, average = 'micro')
   #  test_f1_score = f1_score(y_true, y_pred, average = 'micro')

   #  print(f'Baseline Train Accuracy: {train_accuracy*100 : 0.2f}%')
   #  print(f'Baseline Test Accuracy: {test_accuracy*100 : 0.2f}%')
   #  print(f'Baseline Precision: {test_precision : 0.2f}')
   #  print(f'Baseline Recall: {test_recall : 0.2f}')
   #  print(f'Baseline F1 Score: {test_f1_score : 0.2f}')
   #  print(f'Baseline Confusion Matrix:\n{confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1))}')

    # results['baseline'] = {'test_acc': test_accuracy*100, 'f1': test_f1_score}

    # if SAVE_RESULTS:
    #     ax = plt.figure().gca()
    #     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #     ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #     plt.imshow(confusion_matrix(y_true, y_pred, labels=np.arange(-1*label_idx if label_idx < -1 else -1*label_idx + 1)))
    #     plt.savefig(f'./results/model/{dataset}/cms/baseline.pdf', bbox_inches='tight')

    # Plot results
    fig, ax = plt.subplots(figsize=(18, 4.8))
    x = np.arange(len(results))
    width = 0.4
    ax2 = ax.twinx()

    # ax.bar(x, [results[key][0] for key in results.keys()], color='#4292C6', label='MSE')
    ax.bar(x - width*0.5, [results[key]['test_acc'] for key in ['uncorrupted'] + baseline_models + ['dini']], width, color='#F1C40F', label='Test Accuracy')
    ax2 .bar(x + width*0.5, [results[key]['f1'] for key in ['uncorrupted'] + baseline_models + ['dini']], width, color='#E67E22', label='F1 Score')
    ax.set_ylabel('Test Accuracy (%)')
    ax2.set_ylabel('F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(['Uncorrupted'] + [model.upper() for model in baseline_models] + ['DINI'])
    plt.title(f'Dataset: {dataset.upper()} | Corruption strategy: {args.strategy.upper()}')
    plt.grid(axis='y', linestyle='--')
    plt.savefig(f'./results/model/{dataset}/barplot.pdf', bbox_inches='tight')


