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
from finding_mean_stddev import *
from paths import *
from skimage.io import imread
from skimage.color import gray2rgb


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

#finding the data_val.csv file
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        #print(files)
        if name == 'data_val.csv':
            csv_path = os.path.join(root, name)


val = pd.read_csv(csv_path, sep=',')

test = val.values.tolist()

labels = []
for i in range(len(test)):
    writerid = getname(test[i][0])
    labels.append(writerid)

#print(labels)

#print(test)
train_mean, train_std = [mean, mean, mean], [std, std, std]
transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(resize),
        # transforms.CenterCrop(centercrop),
        #transforms.RandomResizedCrop(randomresizedcrop),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])

resnet = resnet18().cuda()


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
        #output = self.dgmp(output) #dgmp or GeM
        output =  torch.sum(output.view(output.size(0), output.size(1), -1), dim=2) #SPoC

        #########################
        ### Dimension Process ###
        #########################
        output = F.normalize(output, p=2, dim=1) #L2 Normalizartion
        return output

model = Model(model=resnet, dgmp=dgmp)
model.load_state_dict(torch.load(PATH))
model.eval()

encs = np.zeros((len(labels), 512))
#print(len(test))
for i in range(len(test)):
    s = imread(os.path.join(current_path, string, test[i][0]))
    s = gray2rgb(s)
    s = transform(s)
    s = torch.unsqueeze(s, 0)

    output = model.forward(s.cuda())
    tensor = output[0].cpu()
    array = tensor.detach().numpy()
    encs[i] = array

def distances(encs):
    """
    compute pairwise distances, assuming encs is a matrix, where each row contains an l2-normalized encoding
    """
    # compute cosine distance = dot product between l2-normalized
    # descriptors
    dists = 1.0 - encs.dot(encs.T)
    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    #dists.fill_diagonal_(torch.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()
    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[indices[r,k]] == labels[r]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        print('precisions', precisions)
        if len(precisions) == 0:
            precisions = [0]
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
        print('avg_precision',mAP)
    mAP = np.mean(mAP)
    print('mean', mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


evaluate(encs, labels)