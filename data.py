import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from paths import *
from finding_mean_stddev import *
from skimage.io import imread
from skimage.color import gray2rgb


#go to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#finding the data.csv file
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        #print(files)
        if name == 'data.csv':
            csv_path = os.path.join(root, name)


data = pd.read_csv(csv_path, sep=',')
#shufle
data = data.sample(frac=1).reset_index(drop=True)

train_test = data[:-1000] #for train and test in train.py
val = data[-1000:] #for validation in test.py

val.to_csv('data_val.csv', index = False, header = True) #write val data to csv

train, test = train_test_split(train_test, train_size=0.85, random_state=3)


#train_x1, train_x2 are the 2 sets of imagenames in data.csv. train_y is label (similarity)
train_x1 = train.iloc[:, 0]
train_x2 = train.iloc[:, 1]
train_y = train.iloc[:, 2]

#similar for tests
test_x1 = test.iloc[:, 0]
test_x2 = test.iloc[:, 1]
test_y = test.iloc[:, 2]

trainlist = []
for i in range(len(train_x1)):
    sublist = []
    #print(train_x1.iloc[i])
    sublist.append(train_x1.iloc[i])
    sublist.append(train_x2.iloc[i])
    sublist.append(train_y.iloc[i])
    sublist = tuple(sublist)
    trainlist.append(sublist)

testlist = []
for i in range(len(test_x1)):
    sublist = []
    #print(train_x1.iloc[i])
    sublist.append(test_x1.iloc[i])
    sublist.append(test_x2.iloc[i])
    sublist.append(test_y.iloc[i])
    sublist = tuple(sublist)
    testlist.append(sublist)


train_mean, train_std = [mean, mean, mean], [std, std, std]
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Resize(resize),
        #transforms.CenterCrop(centercrop),
        #transforms.RandomResizedCrop(randomresizedcrop),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])


class CustomDataset(Dataset):
    def __init__(self, samples, transform = None):
        self.transform = transform
        self.samples = samples

    def __getitem__(self, index):
        s1, s2, t = self.samples[index]
        s1 = imread(os.path.join(current_path, string, s1))
        s2 = imread(os.path.join(current_path, string, s2))
        s1 = gray2rgb(s1)
        s2 = gray2rgb(s2)

        if self.transform:
            s1 = self.transform(s1)
            s2 = self.transform(s2)

        t = torch.tensor(t)
        return (s1, s2, t)

    def __len__(self):
        return len(self.samples)


train_dataset = CustomDataset(trainlist, transform=transform)
test_dataset = CustomDataset(testlist, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

print('Making train_dataloader and test_dataloader is done')

