import numpy as np
from src.utils import *
from src.folderconstants import *
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sko.GA import GA
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm

def load_data(dataset):
    inp = np.load(f'{output_folder}/{dataset}/inp.npy')
    out = np.load(f'{output_folder}/{dataset}/out.npy')
    inp_c = np.load(f'{output_folder}/{dataset}/inp_c.npy')
    out_c = np.load(f'{output_folder}/{dataset}/out_c.npy')
    data = np.concatenate([inp, out], axis=1)
    data_c = np.concatenate([inp_c, out_c], axis=1)
    return data, data_c

def init_impute(data_c, data_m, strategy = 'zero'):
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
    from src.parser import *
    data, data_c = load_data(args.dataset)
    data_m = np.isnan(data_c)
    data_c = init_impute(data_c, data_m, strategy = 'zero')
    subset = correct_subset(data_c, data_m)
    gm = GaussianMixture(n_components=50, random_state=0).fit(subset)

    print('Starting MSE', mse(data[data_m], data_c[data_m]))
    
    data_new = opt(gm, data_c, data_m)
    print('Final MSE', mse(data[data_m], data_new[data_m]))