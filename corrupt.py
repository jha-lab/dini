import os
import pandas as pd
import numpy as np
import torch
from src.folderconstants import *
from src.corrupt_parser import *
from src.corrupt_utils import *
from matplotlib import pyplot as plt


VISUALIZE = False


def MCAR(df, fraction = 0.1):
	"""Missing Completely At Random"""
	df2 = df.copy(deep=True)
	size = df.values.shape[1]*df.values.shape[0]
	indices = np.random.choice(size, replace=False,
						   size=int(size * fraction))
	df2.values[np.unravel_index(indices, df.values.shape)] = None
	return df2

def MAR(df, fraction = 0.1, p_obs = 0.5):
	"""Missing At Random"""
	df2 = df.copy(deep=True)
	mask = MAR_mask(df.values, fraction, p_obs).double()
	df2.values[mask.bool()] = None
	return df2

def MNAR(df, fraction = 0.1, p_obs = 0.5, opt = "selfmasked", q = 0.3):
	"""Missing Not At Random"""
	df2 = df.copy(deep=True)
	if opt == "logistic":
		mask = MNAR_mask_logistic(df.values, fraction, p_obs).double()
	elif opt == "quantile":
		mask = MNAR_mask_quantiles(df.values, fraction, q, 1 - p_obs).double()
	elif opt == "selfmasked":
		mask = MNAR_self_mask_logistic(df.values, fraction).double()
	df2.values[mask.bool()] = None
	return df2

def MPAR(df, fraction = 0.1, patch_size = 5):
	"""Missing Patches At Random"""
	df2 = df.copy(deep=True)
	patch_size = min(patch_size, min(df.values.shape[0], df.values.shape[1]))
	size = (df.values.shape[0] - patch_size) * (df.values.shape[1] - patch_size)
	indices = np.random.choice(size, replace=False,
						   size=int(size * fraction / (patch_size * patch_size)))
	indices = np.unravel_index(indices, (df.values.shape[0] - patch_size, df.values.shape[1] - patch_size))
	patch_indices = []
	for i in range(len(indices[0])):
		for pi in range(patch_size):
			for pj in range(patch_size):
				patch_indices.append((indices[0][i] + pi, indices[1][i] + pj))
	for idx in patch_indices:
		df2.values[idx] = None
	return df2

def MSAR(df, fraction = 0.1, stream_size = 10):
	"""Missing Streams At Random"""
	df2 = df.copy(deep=True)
	stream_size = min(stream_size, min(df.values.shape[0], df.values.shape[1]))
	size = (df.values.shape[0] - stream_size) * df.values.shape[1]
	indices = np.random.choice(size, replace=False,
						   size=int(size * fraction / stream_size))
	indices = np.unravel_index(indices, (df.values.shape[0] - stream_size, df.values.shape[1]))
	stream_indices = []
	for i in range(len(indices[0])):
		for si in range(stream_size):
			stream_indices.append((indices[0][i] + si, indices[1][i]))
	for idx in stream_indices:
		df2.values[idx] = None
	return df2

def normalize(df):
	return (df-df.min())/(df.max()-df.min())

def visualize(df, dataset, corruption, fraction):
	plt.imshow(df, aspect='auto')
	plt.colorbar()
	plt.xticks(range(df.values.shape[1]), df.columns, rotation=45)
	fraction_str = str(args.fraction).split(".")[-1]
	plt.title(f'Dataset: {dataset.upper()} | Corruption strategy: {corruption.upper()} | Fraction: {fraction}')
	plt.savefig(f'./heatmaps/heatmap_{dataset.lower()}_{corruption.lower()}_p{fraction_str}.pdf', bbox_inches='tight')

def process(dataset, corruption, fraction = 0.1):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	data_file = f'{data_folder}/{dataset}/data.csv'
	df = pd.read_csv(data_file, index_col=0, nrows=1000 if not VISUALIZE else 200)
	assert not np.any(np.isnan(df.values))
	df = normalize(df)
	if corruption == 'MCAR':
		corrupt_df = MCAR(df, fraction)
	elif corruption == 'MAR':
		corrupt_df = MAR(df, fraction)
	elif corruption == 'MNAR':
		corrupt_df = MNAR(df, fraction)
	elif corruption == 'MPAR':
		corrupt_df = MPAR(df, fraction)
	elif corruption == 'MSAR':
		corrupt_df = MSAR(df, fraction)
	else:
		raise NotImplementedError()
	if VISUALIZE: visualize(corrupt_df, dataset, corruption, fraction)
	if dataset == 'MSDS':
		def split(df):
			inp_col = [col for col in df.columns if 'cpu' in col]
			out_col = [col for col in df.columns if 'mem' in col]
			return df[inp_col].values, df[out_col].values
	elif dataset == 'concrete':
		def split(df):
			return df.iloc[:, :-1].values, df.iloc[:, -1].values.reshape(-1, 1)
	elif dataset == 'energy':
		def split(df):
			return df.iloc[:, :-2].values, df.iloc[:, -2:].values
	elif dataset == 'housing':
		def split(df):
			return df.iloc[:, :-1].values, df.iloc[:, -1].values.reshape(-1, 1)
	elif dataset == 'naval':
		def split(df):
			return df.iloc[:, :-2].values, df.iloc[:, -2:].values
	elif dataset == 'diabetes':
		def split(df):
			return df.iloc[:, :-7].values, df.iloc[:, -7:].values
	elif dataset == 'accelerometer':
		def split(df):
			return df.iloc[:, 0:2].values, df.iloc[:, 2:].values
	elif dataset == 'air_quality':
		def split(df):
			return df.iloc[:, 0:-3].values, df.iloc[:, -3:].values
	elif dataset == 'linear':
		def split(df):
			return df.iloc[:, :-5].values, df.iloc[:, -5:].values
	elif dataset == 'diamonds':
		def split(df):
			return df.iloc[:, :-2].values, df.iloc[:, -2:].values
	elif dataset == 'flights':
		def split(df):
			return df.iloc[:, :-5].values, df.iloc[:, -5:].values
	elif dataset == 'traffic':
		def split(df):
			return df.iloc[:, :-1].values, df.iloc[:, -1:].values.reshape(-1, 1)
	elif dataset == 'yacht':
		def split(df):
			return df.iloc[:, :-1].values, df.iloc[:, -1:].values.reshape(-1, 1)
	elif dataset == 'breast':
		def split(df):
			return df.iloc[:, :-1].values, df.iloc[:, -1:].values.reshape(-1, 1)
	elif dataset == 'gas':
		def split(df):
			return df.iloc[:, :-1].values, df.iloc[:, -1:].values.reshape(-1, 1)
	elif dataset == 'swat':
		def split(df):
			return df.iloc[:, :-1].values, df.iloc[:, -1:].values.reshape(-1, 1)
	inp, out = split(df)
	assert not np.any(np.isnan(inp)) and not np.any(np.isnan(out))
	inp_c, out_c = split(corrupt_df)
	for file in ['inp', 'out', 'inp_c', 'out_c']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))

if __name__ == '__main__':
	process(args.dataset, args.strategy)
