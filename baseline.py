import numpy as np
from src.parser import *
from src.utils import *
from src.folderconstants import *
import matplotlib.pyplot as plt
from fancyimpute import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tqdm import tqdm
from copy import deepcopy

def load_data(dataset):
    inp = np.load(f'{output_folder}/{dataset}/inp.npy')
    out = np.load(f'{output_folder}/{dataset}/out.npy')
    inp_c = np.load(f'{output_folder}/{dataset}/inp_c.npy')
    out_c = np.load(f'{output_folder}/{dataset}/out_c.npy')
    data = np.concatenate([inp, out], axis=1)
    data_c = np.concatenate([inp_c, out_c], axis=1)
    return data, data_c

def init_impute(data_c, data_m, strategy = 'zero'):
    data_c = deepcopy(data_c)
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

def opt(gm, dataset, dataset_m):
    new_dataset = []
    for i in tqdm(range(dataset.shape[0]), ncols=80):
        d, d_m = dataset[i], dataset_m[i]
        if not np.any(d_m):
            new_dataset.append(d)
            continue
        inp_size = np.sum(d_m + 0)
        inp = np.zeros(inp_size)
        def fn(i):
            gmm_input = deepcopy(d)
            gmm_input[d_m] = i
            return -gm.score_samples([gmm_input])[0]
        ga = minimize(fn, inp, method='L-BFGS-B', 
            bounds=[(0, 1)]*inp_size)
        best_x = ga.x
        gmm_input = deepcopy(d)
        gmm_input[d_m] = best_x
        new_dataset.append(gmm_input)
    return np.array(new_dataset)
 
if __name__ == '__main__':
    data, data_c = load_data(args.dataset)
    data_m = np.isnan(data_c)
    
    data_c_imputed = init_impute(data_c, data_m, strategy = 'zero')
    # subset = correct_subset(data_c, data_m)

    print('Starting MSE', mse(data[data_m], data_c_imputed[data_m]))

    if args.model == 'mean':
        data_new = SimpleFill(fill_method='mean').fit_transform(data_c)
    elif args.model == 'median':
        data_new = SimpleFill(fill_method='median').fit_transform(data_c)
    elif args.model == 'knn':
        k = [3,10,50][0]
        data_new = KNN(k=k, verbose=False).fit_transform(data_c)
    elif args.model == 'svd':
        rank = [np.ceil((data_c.shape[1]-1)/10),np.ceil((data_c.shape[1]-1)/5), data_c.shape[1]-1][0]
        data_new = IterativeSVD(rank=int(rank), verbose=False).fit_transform(data_c)
    elif args.model == 'mice':
        max_iter = [3,10,50][0]
        data_new = IterativeImputer(max_iter=max_iter).fit_transform(data_c)
    elif args.model == 'spectral':
        sparsity = [0.5,None,3][0]
        data_new = SoftImpute(shrinkage_value=sparsity).fit_transform(data_c)
    elif args.model == 'matrix':
        data_new = MatrixFactorization(verbose=False).fit_transform(data_c)
    else:
        raise NotImplementedError()

    print('Final MSE', mse(data[data_m], data_new[data_m]))
    print('Final MAE', mae(data[data_m], data_new[data_m]))