import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import sys


# Dataset source: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia?resource=download-directory

# Read dataset files

data_dir = './train'

if not os.path.exists(data_dir):
	raise RuntimeError('COVID Chest X-Ray dataset not found. Download from source.')	

model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    img = Image.open(image_name)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    my_embedding = torch.zeros(512)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))

    h = layer.register_forward_hook(copy_data)

    model(t_img)
    h.remove()

    return my_embedding

x, y = [], []
for main_dir_name in ['./train', './test']:
	for dir_name in os.listdir(main_dir_name):
		if not dir_name.startswith('.'):
			for image_name in tqdm(os.listdir(os.path.join(main_dir_name, dir_name)), desc=f'Processing {dir_name} images', ncols=80):
				try:
					vector = get_vector(os.path.join(main_dir_name, dir_name, image_name))
					y_img = 1 if dir_name == 'COVID19' else 0
					x.append(vector.detach().numpy()); y.append(y_img)
				except:
					# print(f'Coundn\'t process image: {image_name}')
					pass

# Process data and create dataframe

x, y = np.array(x), np.array(y).reshape(-1, 1)
data = np.concatenate((x, y), axis=1)

df = pd.DataFrame(data=data, index=None)

df = df.dropna()
df = df.sample(frac=1)

# Save dataframe

df.to_csv('./data.csv', index=False)


