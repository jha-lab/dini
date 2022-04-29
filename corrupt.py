import os
import pandas as pd
import numpy as np
import torch
from src.folderconstants import *
from src.corrupt_parser import *
from src.corrupt_utils import *

datasets = ['MSDS']

def MCAR(df, fraction = 0.1):
	df2 = df.copy(deep=True)
	size = df.values.shape[1]*df.values.shape[0]
	indices = np.random.choice(size, replace=False,
						   size=int(size * fraction))
	df2.values[np.unravel_index(indices, df.values.shape)] = None
	return df2

def MAR(df, fraction = 0.3, p_obs = 0.5):
	df2 = df.copy(deep=True)
	mask = MAR_mask(df.values, fraction, p_obs).double()
	df2.values[mask.bool()] = None
	return df2

def MNAR(df, fraction = 0.1, p_obs = 0.5, opt = "logistic", q = None):
	df2 = df.copy(deep=True)
	if opt == "logistic":
		mask = MNAR_mask_logistic(df.values, fraction, p_obs).double()
	elif opt == "quantile":
		mask = MNAR_mask_quantiles(df.values.double(), fraction, q, 1 - p_obs).double()
	elif opt == "selfmasked":
		mask = MNAR_self_mask_logistic(df.values.double(), fraction).double()
	df2.values[mask.bool()] = None
	return df2

def normalize(df):
	return (df-df.min())/(df.max()-df.min())

def process(dataset, corruption):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	data_file = f'{data_folder}/{dataset}/data.csv'
	df = pd.read_csv(data_file, index_col=0, nrows=1000)
	df = normalize(df)
	if corruption == 'MCAR':
		corrupt_df = MCAR(df)
	elif corruption == 'MAR':
		corrupt_df = MAR(df)
	elif corruption == 'MNAR':
		corrupt_df = MNAR(df)
	else:
		raise NotImplementedError()
	if dataset == 'MSDS':
		def split(df):
			inp_col = [col for col in df.columns if 'cpu' in col]
			out_col = [col for col in df.columns if 'mem' in col]
			return df[inp_col].values, df[out_col].values
	elif dataset == 'concrete':
		def split(df):
			return df.iloc[:, :-1].values, df.iloc[:, -1].values.reshape(-1, 1)
	elif dataset == 'linear':
		def split(df):
			return df.iloc[:, :-5].values, df.iloc[:, -5:].values
	inp, out = split(df)
	inp_c, out_c = split(corrupt_df)
	for file in ['inp', 'out', 'inp_c', 'out_c']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))

if __name__ == '__main__':
	process(args.dataset, args.strategy)
