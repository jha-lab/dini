import pandas as pd
import numpy as np
import os


# Dataset source: https://www.kaggle.com/datasets/shivam2503/diamonds

# Read csv file

df = pd.read_csv('./diamonds.csv')

# Process dataframe

cut = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
clarity = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

df['cut'] = df['cut'].map(cut)
df['color'] = df['color'].map(lambda x: ord(x) - 65)
df['clarity'] = df['clarity'].map(clarity)

df = df[['cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'carat', 'price']]

# Save dataframe

df.to_csv('./data.csv')


