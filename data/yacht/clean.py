import pandas as pd
import numpy as np
import os


# Dataset source: https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics

# Read csv file
	
df = pd.read_csv('./yacht_hydrodynamics.data', sep=r'\s+')

# Process dataframe

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

df = df.dropna()

# Save dataframe

df.to_csv('./data.csv', index=False)


