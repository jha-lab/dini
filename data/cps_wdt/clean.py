import pandas as pd
import numpy as np
import os


# Dataset source: https://ieee-dataport.org/open-access/hardware-loop-water-distribution-testbed-wdt-dataset-cyber-physical-security-testing

# Read csv file

df1 = pd.read_csv('./phy_att_1.csv')
df2 = pd.read_csv('./phy_att_2.csv')
df3 = pd.read_csv('./phy_att_3.csv')

# Process dataframe

df1['Label_n'] = df1['Label_n']*1
df1['Label'] = df1['Label'].map({'normal': 0, 'physical fault': 1, 'MITM': 2, 'DoS': 3, 'scan': 4})

df2['Label_n'] = df2['Label_n']*2
df2['Label'] = df2['Label'].map({'normal': 0, 'physical fault': 1, 'MITM': 2, 'DoS': 3, 'scan': 4})

df3['Label_n'] = df3['Label_n']*3
df3['Label'] = df3['Label'].map({'normal': 0, 'physical fault': 1, 'MITM': 2, 'DoS': 3, 'scan': 4})

df = pd.concat([df1, df2, df3])

df = df.dropna()

df = df.iloc[:, 1:]
df = df*1

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

# Save dataframe

df.to_csv('./data.csv', index=False)


