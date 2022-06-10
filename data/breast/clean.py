import pandas as pd
import numpy as np
import os


# Dataset source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Prognostic%29

# Read csv file
	
df = pd.read_csv('./breast-cancer-wisconsin.data', sep=r',', header=None)

# Process dataframe

df = df.iloc[:, 1:]
nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

df.drop(df.index[df[6] == '?'], inplace=True)
df = df.dropna()

# Save dataframe

df.to_csv('./data.csv', index=False)


