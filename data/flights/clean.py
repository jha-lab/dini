import pandas as pd
import numpy as np
import os


# Dataset source: https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv

# Read csv file

df = pd.read_csv('./flights.csv', low_memory=False)

# Process dataframe

airline = {val: idx for (idx, val) in enumerate(list(set(df['AIRLINE'])))}
tail_number = {val: idx for (idx, val) in enumerate(list(set(df['TAIL_NUMBER'])))}
origin_airport = {val: idx for (idx, val) in enumerate(list(set(df['ORIGIN_AIRPORT'])))}
destination_airport = {val: idx for (idx, val) in enumerate(list(set(df['DESTINATION_AIRPORT'])))}

df['AIRLINE'] = df['AIRLINE'].map(airline)
df['TAIL_NUMBER'] = df['TAIL_NUMBER'].map(tail_number)
df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].map(origin_airport)
df['DESTINATION_AIRPORT'] = df['DESTINATION_AIRPORT'].map(destination_airport)

df = df.fillna(method='ffill')
df = df.iloc[:10000, :-6]
df = df.iloc[:, 3:]

cols = df.columns.tolist()
cols.remove('DEPARTURE_DELAY'); cols.remove('ELAPSED_TIME'); cols.remove('ARRIVAL_DELAY')
cols.extend(['DEPARTURE_DELAY', 'ELAPSED_TIME', 'ARRIVAL_DELAY'])
df = df[cols]

# Save dataframe

df.to_csv('./data.csv', index=False)


