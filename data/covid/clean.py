import pandas as pd
import numpy as np
import os


# Dataset source: S. Hassantabar et al., "CovidDeep: SARS-CoV-2/COVID-19 Test Based on Wearable Medical Sensors and Efficient Neural Networks," in IEEE Transactions on Consumer Electronics, vol. 67, no. 4, pp. 244-256, Nov. 2021, doi: 10.1109/TCE.2021.3130228.

# Read dataset files

data_dir = 'Three-way classification real data'

if not os.path.exists(data_dir):
	raise RuntimeError('CovidDeep dataset not found. Request access from authors and add corresponding folder to this directory.')	

x_train = np.load(os.path.join(data_dir, 'X_train_total.npy'))
x_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train_total.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

# Process data and create dataframe

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
y[y == 2] = 1
data = np.concatenate((x, y), axis=1)

df = pd.DataFrame(data=data, index=None)

df = df.dropna()
df = df.sample(frac=0.1)

# Save dataframe

df.to_csv('./data.csv', index=False)


