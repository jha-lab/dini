import pandas as pd
import numpy as np
import os


# Dataset source: https://ieee-dataport.org/open-access/hardware-loop-water-distribution-testbed-wdt-dataset-cyber-physical-security-testing

# Read csv file

df1 = pd.read_csv('./phy_att_1.csv')
df2 = pd.read_csv('./phy_att_2.csv')
df3 = pd.read_csv('./phy_att_3.csv')

# Process dataframe

df1['Label'] = df1['Label'].map({'normal': 'N', 'physical fault': 'PF', 'MITM': 'MITM', 'DoS': 'DoS', 'scan': 'S'})

df2['Label'] = df2['Label'].map({'normal': 'N', 'physical fault': 'PF', 'MITM': 'MITM', 'DoS': 'DoS', 'scan': 'S'})

df3['Label'] = df3['Label'].map({'normal': 'N', 'physical fault': 'PF', 'MITM': 'MITM', 'DoS': 'DoS', 'scan': 'S'})

df = pd.concat([df1, df2, df3])

df = df.drop(['Label_n'], axis=1)
df = df.dropna()

df = pd.concat([df, pd.get_dummies(df.Label, prefix='Label')], axis=1)
df = df.drop(['Label'], axis=1)

df = df.iloc[:, 1:]
df = df*1

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

# Save dataframe

df.to_csv('./data.csv', index=False)


