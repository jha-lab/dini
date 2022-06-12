import pandas as pd
import numpy as np
import os


# Dataset source: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

# Read csv file

if not os.path.exists('./SWaT_Dataset_Attack_v0.csv'): 
	raise RuntimeError('Raw dataset file not found. Request access from: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/')
	
df = pd.read_csv('./SWaT_Dataset_Attack_v0.csv')

# Process dataframe

df = df.iloc[:, 1:]
df['Normal/Attack'] = df['Normal/Attack'].map({'Normal': 0, 'Attack': 1})
df = df.sample(frac=0.5, random_state=0)

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

df = df.dropna()

# Save dataframe

df.to_csv('./data.csv', index=False)


