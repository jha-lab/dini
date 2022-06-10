import pandas as pd
import numpy as np
import os


# Dataset source: https://archive.ics.uci.edu/ml/datasets/Behavior+of+the+urban+traffic+of+the+city+of+Sao+Paulo+in+Brazil

# Read csv file
	
df = pd.read_csv('./traffic.csv', sep=';', decimal=',')

# Process dataframe

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

df = df.dropna()

# Save dataframe

df.to_csv('./data.csv', index=False)


