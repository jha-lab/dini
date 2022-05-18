import pandas as pd
import numpy as np
import os


# Dataset source: https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv

# Read csv file

df = pd.read_csv('./data.txt', sep='\s+')

# Process dataframe

df = df.dropna()

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

# Save dataframe

df.to_csv('./data.csv', index=False)


