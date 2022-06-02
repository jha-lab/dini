import pandas as pd
import numpy as np
import os


# Dataset source: https://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures

# Read csv file

df1 = pd.read_csv('./ethylene_CO.txt', sep=r'\s+')
df2 = pd.read_csv('./ethylene_methane.txt', sep=r'\s+')

# Process dataframe

df1_cols = df1.columns.tolist()
df2_cols = df2.columns.tolist()

df1_cols = df1_cols[1:]
# df1_cols = df1_cols[2:] + [df1_cols[1],] + [df1_cols[0],] # Sensor data + Ethylene + CO
df1_cols = df1_cols[2:] # Sensor data
df1 = df1[df1_cols]
# df1.insert(18, 'Methane', 0)
df1.insert(16, 'Gas', 0) # Gas -- 0: CO, 1: Methane

df2_cols = df2_cols[1:]
# df2_cols = df2_cols[2:] + [df2_cols[1],] + [df2_cols[0],] # Sensor data + Ethylene + Methane
df2_cols = df2_cols[2:] # Sensor data
df2 = df2[df2_cols]
# df2.insert(17, 'CO', 0)
df2.insert(16, 'Gas', 1) # Gas -- 0: CO, 1: Methane

print(df1.head)
print(df2.head)

df = pd.concat([df1, df2])

df = df.sample(frac=0.07, random_state=0)

# Save dataframe

df.to_csv('./data.csv', index=False)


