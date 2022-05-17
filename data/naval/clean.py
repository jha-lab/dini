import pandas as pd
import numpy as np
import os


# Dataset source: https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv

# Read csv file

column_names = ['lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'T48', 'T1', 'T2', 'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf', 'GTCdsc', 'GTTdcs']

df = pd.read_csv('./data.txt', sep='\s+', header=None, names=column_names)

# Process dataframe

df = df.dropna()

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

# Save dataframe

df.to_csv('./data.csv', index=False)


