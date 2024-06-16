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
from torchvision.models import vgg16

# %%
import torch.optim as optim


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
data_file = "dataset/dataset/train"

# %%
# using vgg16 because number of variables in vgg19 are very large and it is taking too much time to train and also giving memory error
# so although I am using vgg16 but I have written vggg19 in the code so that it can be easily changed to vgg19
class SiameseNN(nn.Module):
    def __init__(self):

        super(SiameseNN, self).__init__()
        self.conolution = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 3),
        )

    def forward_once(self, x):
        output = self.conolution(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, inp1):
        output = self.forward_once(inp1)
        return output

# %%
def check(path1, path2):
    if (path1[22:].split('/')[0]==path2[22:].split('/')[0]):
        return True
    return False


# %%
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    loss = torch.clamp(margin + distance_positive - distance_negative, min=0.0)
    return loss.mean()

# %%
def train(img_pairs, model, loss_fn, optimizer, batch_size, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        random.shuffle(img_pairs)
        for i in range(0, len(img_pairs), batch_size):
            batch = img_pairs[i:i+batch_size]
            anchor_images = []
            positive_images = []
            negative_images = []
            for pair in batch:
                anchor_path, positive_path = pair
                anchor_image = Image.open(anchor_path).convert("L")
                positive_image = Image.open(positive_path).convert("L")
                negative_path = random.choice(img_pairs)[0]
                while check(anchor_path, negative_path):
                    negative_path = random.choice(img_pairs)[0]
                negative_image = Image.open(negative_path).convert("L")
                anchor_images.append(T.ToTensor()(anchor_image))
                positive_images.append(T.ToTensor()(positive_image))
                negative_images.append(T.ToTensor()(negative_image))
            
            anchor_images = torch.stack(anchor_images).to(device)
            positive_images = torch.stack(positive_images).to(device)
            negative_images = torch.stack(negative_images).to(device)

            optimizer.zero_grad()
            anchor_embeddings = model(anchor_images)
            positive_embeddings = model(positive_images)
            negative_embeddings = model(negative_images)

            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            if (i+batch_size) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+batch_size}/{len(img_pairs)}], Loss: {loss.item()}")

# %%
def test(img_pair, label, model):
    model.eval()
    anchor_path, test_path = img_pair
    anchor_image = Image.open(anchor_path).convert("L")
    test_image = Image.open(test_path).convert("L")
    anchor_tensor = T.ToTensor()(anchor_image).unsqueeze(0).to(device)
    test_tensor = T.ToTensor()(test_image).unsqueeze(0).to(device)

    anchor_embedding = model(anchor_tensor)
    test_embedding = model(test_tensor)

    distance = F.pairwise_distance(anchor_embedding, test_embedding)
    if (label == 1 and distance < 1.0) or (label == 0 and distance >= 1.0):
        return True
    return False


# %%
model = SiameseNN().to(device)

# %%
# total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
total_params

# %%
img_pairs = []
anchors = dict()

# %%
for fld in os.listdir(data_file):
    img_set = os.listdir(data_file+"/"+fld)
    anchors[fld] = data_file+"/"+fld+"/"+img_set[0]
    for i in range(len(img_set)):
        for j in range(i+1, len(img_set)):
            img = img_set[i]
            img2 = img_set[j]
            if (img!=img2):
                img_pairs.append([data_file+"/"+fld+"/"+img, data_file+"/"+fld+"/"+img2])


# %%
batch_size = 12
num_epochs = 10
learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = triplet_loss

train(img_pairs, model, loss_fn, optimizer, batch_size, num_epochs)


# %%
torch.save(model, "model.pt")

# %%
torch.save(model.state_dict(), "model.pth")

# %%


# %%
correct = 0
total = 0
for img_pair in img_pairs:
    if (test(img_pair, 1, model)):
        correct += 1
    total += 1

# %%
ind = 0
correct = 0
total = 0
for img_pair in img_pairs:
    ind += 1
    if (ind==10000):
        break
    img1_path, _ = img_pair
    img2_path = random.choice(img_pairs)[0]
    while(check(img1_path, img2_path)):
        img2_path = random.choice(img_pairs)[0]
    if (test([img1_path, img2_path], 0, model)):
        correct += 1
    total += 1

# %%
correct

# %%
total
