from paths import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
from PIL import Image

from torchvision import transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#go to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
transform = transforms.Compose([
        #transforms.Resize(resize),
        transforms.CenterCrop(centercrop),
        transforms.ToTensor()
    ])


#finding the data.csv file
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        #print(files)
        if name == 'images.csv':
            csv_path = os.path.join(root, name)


data = pd.read_csv(csv_path, sep=',')
data = data.values.tolist()


class MyDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        s = self.samples[index]
        s = Image.open(os.path.join(current_path, string, s[0]))
        s = self.transform(s)
        return s

    def __len__(self):
        return len(self.samples)


dataset = MyDataset(data, transform= transform)
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)

mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print("Getting mean and standard deviation done", mean, std)