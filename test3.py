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
from dgmp import *
from model import *
import torch.nn.functional as F
from loss import ContrastiveLoss
from sklearn import decomposition



#go to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#finding the data_test.csv file
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        #print(files)
        if name == 'data_test.csv':
            csv_path = os.path.join(root, name)


data = pd.read_csv(csv_path, sep=',')
print('total', len(data))

test_x1 = data.iloc[:, 0]
test_x2 = data.iloc[:, 1]
test_y = data.iloc[:, 2]

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
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.Resize(400),
        transforms.CenterCrop(400),
        #transforms.RandomResizedCrop(400),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])

class CustomDataset(Dataset):
    def __init__(self, samples, transform = None):
        self.transform = transform
        self.samples = samples

    def __getitem__(self, index):
        s1, s2, t = self.samples[index]
        s1 = imread(os.path.join(current_path, string_test, s1))
        s2 = imread(os.path.join(current_path, string_test, s2))
        s1 = gray2rgb(s1)
        s2 = gray2rgb(s2)

        if self.transform:
            s1 = self.transform(s1)
            s2 = self.transform(s2)

        t = torch.tensor(t)
        return (s1, s2, t)

    def __len__(self):
        return len(self.samples)

test_dataset = CustomDataset(testlist, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

print('Making test_dataloader is done')

resnet = resnet18().cuda()

pca = decomposition.PCA()


class Model(torch.nn.Module):
    def __init__(self, model):
    #def __init__(self, model, pool):
        super(Model, self).__init__()
        self.model = model
        #self.pool = pool

    def forward(self, sample):
        output = self.model(sample)

        ####################
        ### Aggregation ###
        ####################
        #output = self.pool(output) #dgmp or GeM
        output =  torch.sum(output.view(output.size(0), output.size(1), -1), dim=2) #SPoC
        #alist = [] #crow
        #for i in range(output.shape[0]): #crow
        #    output2 = apply_crow_aggregation(output[i]) #crow
        #    alist.append(output2) #crow
        #output = torch.stack(alist) #crow

        #########################
        ### Dimension Process ###
        #########################
        output = F.normalize(output, p=2, dim=1) #L2 Normalizartion
        return output

model = Model(model=resnet)
#model = Model(model=resnet, pool=gmp)
model.load_state_dict(torch.load(PATH))
model.eval()


criterion = ContrastiveLoss()

true = 0
false = 0
TP, FP, TN, FN = 0,0,0,0
with torch.no_grad():
    for e, (inputs1, inputs2, labels) in enumerate(test_dataloader):
        output1 = model.forward(inputs1.cuda())
        output2 = model.forward(inputs2.cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        #print('euclidean distance', euclidean_distance)
        #print('labels', labels)
        #loss_contrastive = criterion(output1, output2, labels.cuda())
        #print("Validation loss {}\n".format(loss_contrastive.item()))
        for i in range(len(euclidean_distance)):
            #print(euclidean_distance[i] - labels[i])
            if abs(euclidean_distance[i] - labels[i]) < 0.5:
                true = true  + 1
            else:
                false = false + 1

            if labels[i] == 0:
                if abs(euclidean_distance[i]) < 0.5:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                if abs(euclidean_distance[i]) > 0.5:
                    TN = TN + 1
                else:
                    FN = FN + 1

print('accuracy', true/(true + false)*100)
print(false/(true + false)*100)
print('True Positive,', TP, 'False Positive,', FP, 'True Negative,', TN, 'False Negative,', FN)

TPR = TP/(FN + TP)
FPR = FP/(TN + FP)
TNR = TN/(TN + FP)
FNR = FN/(FN + TP)
print('True Positive Rate,', TPR, 'False Positive Rate,', FPR, 'True Negative Rate,', TNR, 'False Negative Rate,', FNR)

accuarcy = (TN + TP)/(TN + TP + FN + FP)
precision = TP/(TP + FP)
print('accuaracy', accuarcy, 'precision', precision)

print('done!')

