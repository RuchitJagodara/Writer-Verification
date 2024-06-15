from torch import nn
import os
import torchvision
from torch.nn import functional as F
import torch
import random
import argparse, random, copy
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

# %%
import torch.optim as optim

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
data_file = "dataset/dataset/train"

# %%
class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(2*self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, inp1, inp2):
        ouput1 = self.forward_once(inp1)
        ouput2 = self.forward_once(inp2)
        output = torch.cat((ouput1, ouput2), 1)
        output = self.fc(output)
        return output

# %%
def check(path1, path2):
    if (path1[22:].split('/')[0]==path2[22:].split('/')[0]):
        return True
    return False

# %%
transform = T.CenterCrop((120, 2000))
ind = 0

def test(img_path_pair, label, model):
    img1_path, img2_path = img_path_pair
    img1 = Image.open(img1_path)
    img1 = transform(img1)
    img2 = Image.open(img2_path)
    img2 = transform(img2)
    label = torch.tensor(label).float().to(device)
    label = label.reshape(1, 1)
    img1 = torch.tensor(np.array(img1)).float().unsqueeze(0).to(device)
    img2 = torch.tensor(np.array(img2)).float().unsqueeze(0).to(device)
    img1 = img1.reshape(1, 1, 120, 2000)
    img2 = img2.reshape(1, 1, 120, 2000)
    output = model(img1, img2)
    if (output>0 and label==1):
        return 1
    elif(output<0 and label==0):
        return 1
    return 0

def train_batch(batch, labels, model, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    batch_size = len(batch)
    img1_batch = []
    img2_batch = []
    for img_pair in batch:
        img1_path, img2_path = img_pair
        img1 = Image.open(img1_path)
        img1 = transform(img1)
        img2 = Image.open(img2_path)
        img2 = transform(img2)
        img1 = torch.tensor(np.array(img1)).float().unsqueeze(0).to(device)
        img2 = torch.tensor(np.array(img2)).float().unsqueeze(0).to(device)
        img1 = img1.reshape(1, 1, 120, 2000)
        img2 = img2.reshape(1, 1, 120, 2000)
        img1_batch.append(img1)
        img2_batch.append(img2)
    img1_batch = torch.cat(img1_batch, dim=0)
    img2_batch = torch.cat(img2_batch, dim=0)
    output = model(img1_batch, img2_batch)
    labels = torch.tensor(labels).float().to(device)
    labels = labels.reshape(batch_size, 1)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def train(img_pairs, model, loss_fn, optimizer, batch_size, num_epochs):
    global ind
    for epoch in range(num_epochs):
        random.shuffle(img_pairs)
        total_loss = 0
        for i in range(0, len(img_pairs), batch_size):
            batch = img_pairs[i:i+batch_size]
            labels = [1] * len(batch)
            for j in range(len(batch)):
                new_img = random.choice(img_pairs)[0]
                while(check(new_img, batch[j][0])):
                    new_img = random.choice(img_pairs)[0]
                batch.append((new_img, batch[j][0]))
                labels.append(0)
            loss = train_batch(batch, labels, model, loss_fn, optimizer)
            total_loss += loss
            ind += 1
            if (ind+1) % 100 == 0:
                print("Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, ind+1, len(img_pairs), loss))
        print("Epoch [{}/{}], Average Loss: {:.4f}".format(epoch+1, num_epochs, total_loss / (len(img_pairs) // batch_size)))

# %%
model = SiameseNN().to(device)

# %%
img_pairs = []

# %%
for fld in os.listdir(data_file):
    img_set = os.listdir(data_file+"/"+fld)
    for img in img_set:
        for img2 in img_set:
            if (img!=img2):
                img_pairs.append([data_file+"/"+fld+"/"+img, data_file+"/"+fld+"/"+img2])

# %%
batch_size = 8
num_epochs = 10
learning_rate = 0.0001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()

train(img_pairs, model, loss_fn, optimizer, batch_size, num_epochs)

# %%
torch.save(model, "model.pt")

# %%
torch.save(model.state_dict(), "model.pth")

# %%


# %%
test(img_pairs[0], 1, model)
