#!/usr/bin/env python3
import os
import csv
import pandas as pd
from paths import *

# empty list to store the images files names
files_list = []

# go to current directory
#os.chdir(os.path.dirname(os.path.abspath(__file__)))

# the directory containing the image files
for i in range(0,len(entries), 3):
    #print(entry)
    files_list.append(entries[i])

#print(files_list)
print('length of image file', len(files_list))


# saving the names of the files in a csv file named images.csv
with open('images.csv', 'w', newline='') as file:
    fieldnames = ['filename']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for i in files_list:
        writer.writerow({'filename': i})

# finding the images.csv file
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        # print(files)
        if name == 'images.csv':
            csv_path = os.path.join(root, name)

data = pd.read_csv(csv_path, sep=',')

#print(len(data))
images = data.iloc[:, 0]

# empty lists to store the file names in iterative way
list1 = []
list2 = []


for j in range(3, annotation_range, 80):
    for i in range(len(images) - j):
        list1.append(images[i])
        list2.append(images[i + j])

#print(list1[:6])
#print(list2[:6])

#print(len(list1), len(list2))

list3 = []


# get the first few letters of the file names that denote writer
def getname(filename):
    string = ''
    for i in filename:
        if i != '-':
            string = string + i
        else:
            break
    return string

#print(list1[0], getname(list1[0]))

for i in range(len(list1)):
    # if same writers => similarity = 0
    if getname(list1[i]) == getname(list2[i]):
        list3.append(0)

    # if different writers => similarity = 1
    else:
        list3.append(1)

# print('list3', len(list3))


# saving the 2 sets of names and similarity of the files in a csv file named data.csv
with open('data.csv', 'w', newline='') as file:
    fieldnames = ['filename1', 'filename2', 'similarity']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(list1)):
        writer.writerow({'filename1': list1[i], 'filename2': list2[i], 'similarity': list3[i]})

print('Annotations done')

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

print('length of data file', len(data))

zeros = 0
ones = 0
for i in range(len(data)):
    if data.iloc[i][2] == 0:
        zeros = zeros + 1
    else:
        ones = ones + 1

print('similar', zeros)
print('dissimilar', ones)