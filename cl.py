import cleanlab
import sklearn
import pandas as pd
from src import models
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
import argparse


def modify_data(y, frac=0.1):
    chosen_ids = np.random.choice(y.shape[0], int(frac * y.shape[0]))

    assert min(y) == 0 and max(y) == 1

    for i in chosen_ids:
        y[i] = 1 - y[i]

    return y, chosen_ids


def ids_to_bool(ids, num_ids):
    ids_bool = []

    for i in range(num_ids):
        if i in ids:
            ids_bool.append(True)
        else:
            ids_bool.append(False)

    return ids_bool


def main(args):
    dataset = 'breast'

    print(f'Testing Confidence Learning on "{dataset}" dataset')

    data_file = f'./data/{dataset}/data.csv'

    # Instantiate dataset
    df = pd.read_csv(data_file, index_col=0, nrows=1000)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1:].values.reshape(-1, 1)
    y = (y - 2)/2; y = y.astype(int).reshape(-1)
    num_ids = y.shape[0]

    # Modify data
    y_mod, chosen_ids = modify_data(y, frac=args.frac)
    chosen_bool = ids_to_bool(chosen_ids, num_ids)

    # Instantiate CL model
    sk_cl = sklearn.linear_model.LogisticRegression()
    cl = cleanlab.classification.CleanLearning(sk_cl)

    label_issues = cl.find_label_issues(X, y_mod)

    # Get indices with issues as per CL
    issue_ids, clean_ids = [], []
    for i in range(label_issues.values.shape[0]):
        if label_issues['is_label_issue'][i] == True:
            # print(f'Label index with issue: {i},\tGiven label: {y_mod[i]}')
            issue_ids.append(i)
        else:
            clean_ids.append(i)

    # Fit CL model
    cl.fit(X, y_mod)

    # # Health summary of entire dataset
    # print(cleanlab.dataset.health_summary(y_mod, confident_joint=cl.confident_joint))

    # Get F1 score for CL
    print(f'CL F1 score: {sklearn.metrics.f1_score(chosen_bool, ids_to_bool(issue_ids, num_ids))}')

    # Instantiate DINI model
    model = models.FCN2(X.shape[1], 1, 512, mc_dropout=True)
    optimizer = torch.optim.Adam(model.parameters() , lr=0.0001, weight_decay=1e-3)
    epoch = -1; num_epochs = 100

    dataloader = DataLoader(list(zip(X, y.astype(float).reshape(-1, 1))), batch_size=1, shuffle=False)
    lf = lambda x, y: torch.sqrt(nn.MSELoss(reduction = 'mean')(x, y)) + nn.L1Loss(reduction = 'mean')(x, y)
    lfo = nn.BCELoss(reduction = 'mean')
    use_ce = True

    # Train DINI model
    e = 0; ls = []
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), ncols=80, desc=f'Training DINI'):
        ls = []
        for inp, out in tqdm(dataloader, leave=False, ncols=80):
            pred_i, pred_o = model(inp.float(), out.float())
            loss = lfo(pred_o.float(), out.float()) if use_ce else lf(pred_o.float(), out.float())
            ls.append(loss.item())   
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # tqdm.write(f'Loss: {np.mean(ls):0.3f}')

    # Get standard deviations on inputs
    inp_std_list, out_std_list = [], []
    for inp, out in tqdm(dataloader, leave=False, ncols=80):
        pred_i_list, pred_o_list = [], []
        for _ in range(50):
            pred_i, pred_o = model(inp.float(), out.float())
            pred_i_list.append(pred_i); pred_o_list.append(pred_o)
        inp_std = torch.std(torch.stack(pred_i_list).squeeze(), dim=0, keepdim=True); out_std = torch.std(torch.stack(pred_o_list).squeeze(), dim=0, keepdim=True)
        inp_std_list.append(inp_std); out_std_list.append(out_std.item())

    # print(out_std_list)
    # print([out_std_list[j] for j in issue_ids])

    issue_ids, clean_ids = [], []
    for i in range(len(out_std_list)):
        if out_std_list[i] >= 0.02: 
            # print(f'DINI label index with issue: {i}')
            issue_ids.append(i)
        else:
            clean_ids.append(i)

    # Get F1 score for CL
    print(f'DINI F1 score: {sklearn.metrics.f1_score(chosen_bool, ids_to_bool(issue_ids, num_ids))}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINI vs. CL')
    parser.add_argument('--frac', 
                        type=float, 
                        required=False,
                        default=0.1,
                        help="fracion of data to modify")
    args = parser.parse_args()

    main(args)