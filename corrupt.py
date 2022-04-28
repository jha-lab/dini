import os
import pandas as pd
import numpy as np
from src.folderconstants import *
from src.corrupt_parser import *

datasets = ['MSDS']

def MCAR(df, fraction = 0.1):
	df2 = df.copy(deep=True)
	size = df.values.shape[1]*df.values.shape[0]
	indices = np.random.choice(size, replace=False,
                           size=int(size * fraction))
	df2.values[np.unravel_index(indices, df.values.shape)] = None
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
	else:
		corrupt_df = MCAR(df)
	if dataset == 'MSDS':
		def split(df):
			inp_col = [col for col in df.columns if 'cpu' in col]
			out_col = [col for col in df.columns if 'mem' in col]
			return df[inp_col].values, df[out_col].values
	inp, out = split(df)
	inp_c, out_c = split(corrupt_df)
	for file in ['inp', 'out', 'inp_c', 'out_c']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))

if __name__ == '__main__':
	process(args.dataset, args.strategy)
