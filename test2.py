import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from torchvision import transforms
from paths import *
from dgmp import dgmp
from model import *
import torch
import torch.nn.functional as F
import os
from data import *
from finding_mean_stddev import *
from paths import *
from skimage.io import imread
from skimage.color import gray2rgb
from loss import ContrastiveLoss
from sklearn import decomposition




# get the first few letters of the file names that denote writer
def getname(filename):
    string = ''
    for i in filename:
        if i != '-':
            string = string + i
        else:
            break
    return string


#go to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#finding the data.csv file
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        #print(files)
        if name == 'data_val.csv':
            csv_path = os.path.join(root, name)


val = pd.read_csv(csv_path, sep=',')

print('total', len(val))
#val_x1, val_x2 are the 2 sets of imagenames in data_val.csv. val_y is label (similarity)
val_x1 = val.iloc[:, 0]
val_x2 = val.iloc[:, 1]
val_y = val.iloc[:, 2]

vallist = []
for i in range(len(val_x1)):
    sublist = []
    #print(train_x1.iloc[i])
    sublist.append(val_x1.iloc[i])
    sublist.append(val_x2.iloc[i])
    sublist.append(val_y.iloc[i])
    sublist = tuple(sublist)
    vallist.append(sublist)

val_mean, val_std = [mean, mean, mean], [std, std, std]
transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        # transforms.Resize(resize),
        # transforms.CenterCrop(centercrop),
        #transforms.RandomResizedCrop(randomresizedcrop),
        transforms.ToTensor(),
        transforms.Normalize(mean=val_mean, std=val_std)
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

val_dataset = CustomDataset(vallist, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4, drop_last=True)

resnet = resnet18().cuda()

pca = decomposition.PCA()


class Model(torch.nn.Module):
    def __init__(self, model, dgmp):
        super(Model, self).__init__()
        self.model = model
        self.dgmp = dgmp

    def forward(self, sample):
        output = self.model(sample)

        ####################
        ### Aggregation ###
        ####################
        output = self.dgmp(output) #dgmp or GeM
        #output =  torch.sum(output.view(output.size(0), output.size(1), -1), dim=2) #SPoC
        #alist = [] #crow
        #for i in range(output.shape[0]): #crow
        #    output2 = apply_crow_aggregation(output[i]) #crow
        #    alist.append(output2) #crow
        #output = torch.stack(alist) #crow

        #########################
        ### Dimension Process ###
        #########################
        #output = F.normalize(output, p=2, dim=1) #L2 Normalizartion
        pca.n_components = 20  #PCA
        output = torch.tensor(pca.fit_transform(output.cpu().detach().numpy()), requires_grad=True).cuda()  #PCA
        return output

model = Model(model=resnet, dgmp=dgmp)
model.load_state_dict(torch.load(PATH))
model.eval()

criterion = ContrastiveLoss()

true = 0
false = 0
TP, FP, TN, FN = 0,0,0,0
with torch.no_grad():
    for e, (inputs1, inputs2, labels) in enumerate(val_dataloader):
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