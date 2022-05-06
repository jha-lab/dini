import numpy as np
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
 
if __name__ == '__main__':
    from src.parser import *
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