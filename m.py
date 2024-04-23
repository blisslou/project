import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


real_dir = "C:\\ml\\training data\\real_and_fake_face\\training_real"

real_path = os.listdir(real_dir)

fake_dir = "C:\\ml\\training data\\real_and_fake_face\\training_fake"

fake_path = os.listdir(fake_dir)

def load_img(path):
    '''Loading images from directory 
    and changing color space from cv2 standard BGR to RGB 
    for better visualization'''
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

fig = plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(load_img(real_dir +"//"+ real_path[i]))
    plt.suptitle("Real faces", fontsize=20)
    plt.axis('off')


fig = plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(load_img(fake_dir +"//"+ fake_path[i]))
    plt.suptitle("Fake faces", fontsize=20)
    plt.axis('off')    

print("hello")    

real_df = pd.DataFrame({'image_path': real_dir +"//"+ real_path[i], 'label': 1} for i in range(0, 1081))
fake_df = pd.DataFrame({'image_path': fake_dir +"//"+ fake_path[i], 'label': 0} for i in range(0, 960))
print("hi")

df = pd.concat([real_df, fake_df], ignore_index=True)
df.tail(10)

df = shuffle(df)
df = df.reset_index(drop=True)
df.head(10)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)



image_size = 224
batch_size = 64
num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device



image_transforms = {'train_transform': A.Compose([A.Resize(image_size, image_size), 
             A.HorizontalFlip(p=0.5), 
                                                  A.RandomBrightnessContrast(always_apply=False, 
                                                                             p=0.4),
                                                  A.Solarize(always_apply=False, 
                                                             p=0.4, 
                                                             threshold=(42, 42)),
                                                  A.MultiplicativeNoise(always_apply=False, 
                                                                        p=0.8, 
                                                                        multiplier=(0.6800000071525574, 1.409999966621399), 
                                                                        per_channel=True, 
                                                                        elementwise=True),
                                                  A.Normalize(mean=(0.485, 0.456, 0.406), 
                                                              std=(0.229, 0.224, 0.225), 
                                                              max_pixel_value=255.0, 
                                                              p=1.0), 
                                                  ToTensorV2()]),
                    
                   'validation_transform': A.Compose([A.Resize(image_size, image_size), 
                                                      A.Normalize(mean=(0.485, 0.456, 0.406), 
                                                                  std=(0.229, 0.224, 0.225), 
                                                                  max_pixel_value=255.0, 
                                                                  p=1.0), 
                                                      ToTensorV2()]),
                   'visualization_transform': A.Compose([A.Resize(image_size, image_size), 
                                                         A.HorizontalFlip(p=0.5), 
                                                         A.RandomBrightnessContrast(always_apply=False, 
                                                                                    p=0.4),
                                                  A.Solarize(always_apply=False, 
                                                             p=0.4, 
                                                             threshold=(42, 42)),
                                                  A.MultiplicativeNoise(always_apply=False, 
                                                                        p=0.8, 
                                                                        multiplier=(0.6800000071525574, 1.409999966621399), 
                                                                        per_channel=True, 
                                                                        elementwise=True)])}


class ImageDataset(Dataset):
    def __init__(self, image_labels, image_dir, transform=None, target_transform=None):
        self.image_labels = image_labels
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        
        
    def __len__(self):
        return len(self.image_labels)
    
    
    def __getitem__(self, index):
        image_path = self.image_dir.iloc[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.image_labels.iloc[index]
        if self.transform:
            image = self.transform(image=image)['image']
        if self.target_transform:
            label = self.target_transform(label=label)
        return image, label

train_label = train_df['label']
train_features = train_df['image_path']

val_label = val_df['label']
val_features = val_df['image_path']

train_dataset = ImageDataset(train_label, 
                             train_features, 
                             transform=image_transforms['train_transform'])
val_dataset = ImageDataset(val_label, 
                           val_features, 
                           transform=image_transforms['validation_transform'])
visual_train_dataset =  ImageDataset(train_label, 
                                     train_features, 
                                     transform=image_transforms['visualization_transform'])  


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
visual_loader = DataLoader(visual_train_dataset, batch_size=batch_size, shuffle=True)

val_loader

visual_train_f, visual_train_t = next(iter(visual_loader))
print(f'Feature batch shape: {visual_train_f.size()}')
print(f'Target batch shape: {visual_train_t.size()}')

for item in visual_loader:
    img, label = item[0], item[1]
    print(img, label)


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.batchnorm_1 = nn.BatchNorm2d(18)
        self.conv_2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=3)
        self.batchnorm_2 = nn.BatchNorm2d(18)
        self.conv_3 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3)
        self.fc_1 = nn.Linear(21632, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.maxpool(nn.functional.relu(self.conv_1(x)))
        x = self.maxpool(nn.functional.relu(self.conv_2(x)))
        x = self.maxpool(nn.functional.relu(self.conv_3(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc_1(x))
        x = nn.functional.relu(self.fc_2(x))
        x = torch.sigmoid(self.classifier(x))
        return x


model_custom = FaceNet()
model_custom.to(device)

criterion_custom = torch.nn.BCELoss()
optimizer_custom = torch.optim.Adam(model_custom.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler_custom = torch.optim.lr_scheduler.ExponentialLR(optimizer_custom, gamma=0.9)

def training_loop(model, training_loader, validation_loader, criterion, optimizer, scheduler, epochs=num_epochs):
    '''Training loop for train and eval modes'''
    for epoch in range(1, epochs+1):
        model.train()
        train_accuracy = 0
        train_loss = 0
        for image, target in training_loader:
            image = image.to(device)
            target = target.to(device)
            target = target.unsqueeze(1)
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(image))
            loss = criterion(outputs.float(), target.float())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_accuracy += ((outputs > 0.5) == target).float().mean().item()
            
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            val_accuracy = 0
            for val_image, val_target in validation_loader:
                val_image = val_image.to(device)
                val_target = val_target.to(device)
                val_target = val_target.unsqueeze(1)
                val_outputs = torch.sigmoid(model(val_image))
                val_loss = criterion(val_outputs.float(), val_target.float())
                
                valid_loss += val_loss.item()
                val_accuracy += ((val_outputs > 0.5) == val_target).float().mean().item() 
                
        print(f'Epoch: {epoch} Train loss: {train_loss/len(training_loader)} Train accuracy: {train_accuracy /len(training_loader)} Val loss: {valid_loss/len(validation_loader)} Val accuracy: {val_accuracy/len(validation_loader)}')
        scheduler.step()



training_loop(model_custom, 
              train_loader, 
              val_loader, 
              criterion_custom, 
              optimizer_custom, 
              scheduler_custom, 
              epochs=num_epochs)


model_np = models.resnet50(pretrained=False)
model_np.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), 
                     nn.ReLU(inplace=True),
                     nn.Linear(in_features=512, out_features=1, bias=True))

model_np.to(device)


criterion_np = torch.nn.BCELoss()
optimizer_np = torch.optim.Adam(model_np.parameters(), lr=0.00001, weight_decay=1e-5)
scheduler_np = torch.optim.lr_scheduler.ExponentialLR(optimizer_np, gamma=0.9)
training_loop(model_np, 
              train_loader, 
              val_loader, 
              criterion_np, 
              optimizer_np, 
              scheduler_np, 
              epochs=num_epochs)

model_p = models.resnet50(pretrained=True)
model_p.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), 
                     nn.ReLU(inplace=True),
                     nn.Linear(in_features=512, out_features=1, bias=True))
for param in model_p.parameters():
    param.requires_grad = True
model_p.to(device)

criterion_p = torch.nn.BCELoss()
optimizer_p = torch.optim.Adam(model_p.parameters(), lr=0.00001, weight_decay=1e-5)
scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optimizer_p, gamma=0.9)
training_loop(model_p, 
              train_loader, 
              val_loader, 
              criterion_p, 
              optimizer_p, 
              scheduler_p, 
              epochs=num_epochs)

   