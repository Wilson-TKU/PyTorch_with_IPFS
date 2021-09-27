#about ipfs...
import ipfshttpclient
import pytorchipfs
# import ipfsApi
import ipfshttpclient

#about torch...
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

#using numpy
import numpy as np

#for data load or save
import pandas as pd

#visualize some datasets
import matplotlib.pyplot as plt

#check our work directory
import os

#to unzip datasets
import zipfile

lr = 0.001 # learning_rate
batch_size = 100 # we will use mini-batch method
epochs = 10 # How much to train a model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(1234)
if device =='cuda':
    torch.cuda.manual_seed_all(1234)


#----------ipfs-----------
# hashes = [
#     'QmVdDq3F4vwhZ3VYshnTfySDMKLQqVbv92TpojBKK7DViF',
#     'QmfYbXadCN3y7BAtPw4VUjb25MxWHMLJZV9vdHuaXDEfmG'
# ]
# hashes = [
#     'bafkreic3aeripksj7a7pnvkiybq3i43hme6pxlmpx7jaokubpz2lfdrvti',
#     'bafybeic7qbuo2ail2y5urbm5btfp7dwcxigjs4kq6m36ecbozaurt4z3te',
#     'bafkreidcct7qpk3tadwtqmboncnmfouu674vusm4zhvuxcmf2n57wxeqfa'
# ]

# client = ipfshttpclient.connect()
# ------CONNECT-----------------------------------
# client.id()
# Standard dataset
# dataset = pytorchipfs.datasets.IPFSImageTensorDataset(
#     client,
#     '../input/dogs-vs-cats-redux-kernels-edition', # Where the files will be downloaded
#     None, # Don't make assumptions about the image shape
#     hashes
# )
# print(dataset)
# api = ipfsApi.Client(host='http://127.0.0.1', port=5001)
# print('1')
# api.cat('QmVdDq3F4vwhZ3VYshnTfySDMKLQqVbv92TpojBKK7DViF')
# client.cat(api['QmVdDq3F4vwhZ3VYshnTfySDMKLQqVbv92TpojBKK7DViF'])
# api.get('QmVdDq3F4vwhZ3VYshnTfySDMKLQqVbv92TpojBKK7DViF')
# print('2')
# api.get('QmfYbXadCN3y7BAtPw4VUjb25MxWHMLJZV9vdHuaXDEfmG')
# print('3')
#----------ipfs-----------
# 11111111
# api = ipfsapi.connect('127.0.0.1', 5001)
# api = ipfsApi.Client(host='https://ipfs.infura.io', port=5001)
# api.id()
# print('1')
# # api.cat('QmVdDq3F4vwhZ3VYshnTfySDMKLQqVbv92TpojBKK7DViF')
# api.cat('bafkreic3aeripksj7a7pnvkiybq3i43hme6pxlmpx7jaokubpz2lfdrvti')
# # api.get('bafkreic3aeripksj7a7pnvkiybq3i43hme6pxlmpx7jaokubpz2lfdrvti')
# print('2')

#22222222



client = ipfshttpclient.connect()
f = open("test.zip", "wb")
f.write(client.get('QmVdDq3F4vwhZ3VYshnTfySDMKLQqVbv92TpojBKK7DViF'))
f.close()
# res = 




#----------ipfs-----------

os.listdir('../input/dogs-vs-cats-redux-kernels-edition')
os.makedirs('../data', exist_ok=True)
base_dir = '../input/dogs-vs-cats-redux-kernels-edition'
train_dir = '../data/i_train'
test_dir = '../data/i_test'

with zipfile.ZipFile(os.path.join(base_dir, 'i_train.zip')) as train_zip:
    train_zip.extractall('../data')
    
with zipfile.ZipFile(os.path.join(base_dir, 'i_test.zip')) as test_zip:
    test_zip.extractall('../data')

os.listdir(train_dir)[:5]

import glob

train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
len(train_list)

from PIL import Image
random_idx = np.random.randint(1,25000,size=10)

fig = plt.figure()
i=1
for idx in random_idx:
    ax = fig.add_subplot(2,5,i)
    img = Image.open(train_list[idx])
    plt.imshow(img)
    i+=1

plt.axis('off')
plt.show()

train_list[0].split('/')[-1].split('.')[0]
int(test_list[0].split('/')[-1].split('.')[0])
print(len(train_list), len(test_list))

from sklearn.model_selection import train_test_split
train_list, val_list = train_test_split(train_list, test_size=0.2)

#data Augumentation
train_transforms =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


test_transforms = transforms.Compose([   
    transforms.Resize((224, 224)),
     transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

class dataset(torch.utils.data.Dataset):
    #가져와서 처리
    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform
        
        
    #dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    #load an one of images
    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label=1
        elif label == 'cat':
            label=0
            
        return img_transformed,label

train_data = dataset(train_list, transform=train_transforms)
test_data = dataset(test_list, transform=test_transforms)
val_data = dataset(val_list, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))
print(len(val_data), len(val_loader))

#check our images shape
train_data[0][0].shape

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = Cnn().to(device)
model.train()

optimizer = optim.Adam(params = model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 10

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        
        output = model(data)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
        
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
    
    
    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            
            val_output = model(data)
            val_loss = criterion(val_output,label)
            
            
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_loader)
            epoch_val_loss += val_loss/ len(val_loader)
            
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))

dog_probs = []
model.eval()
with torch.no_grad():
    for data, fileid in test_loader:
        data = data.to(device)
        preds = model(data)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), preds_list))

dog_probs.sort(key = lambda x : int(x[0]))
dog_probs

idx = list(map(lambda x: x[0],dog_probs))
prob = list(map(lambda x: x[1],dog_probs))

submission = pd.DataFrame({'id':idx,'label':prob})
submission
submission.to_csv('result.csv',index=False)

# Check model performance and visualize some data
import random

id_list = []
class_ = {0: 'cat', 1: 'dog'}

fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

for ax in axes.ravel():
    
    i = random.choice(submission['id'].values)
    
    label = submission.loc[submission['id'] == i, 'label'].values[0]
    if label > 0.5:
        label = 1
    else:
        label = 0
        
    img_path = os.path.join(test_dir, '{}.jpg'.format(i))
    img = Image.open(img_path)
    
    ax.set_title(class_[label])
    ax.imshow(img)

