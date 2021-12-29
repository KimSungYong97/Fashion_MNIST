import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

import data
import CNN
import training
import accuracy
import confusion_mat
import visualize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_csv=pd.read_csv("./data/fashion_mnist_data/fashion-mnist_train.csv")
test_csv=pd.read_csv("./data/fashion_mnist_data/fashion-mnist_test.csv")

train_set = data.FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = data.FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(test_set, batch_size=100)


model = CNN.FashionCNN()
model.to(device)

error = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

#------Training a network and Testing it on test dataset------


num_epochs = 5
loss_list = []
iteration_list = []
accuracy_list = []
predictions_list = []
labels_list = []

loss_list,iteration_list,accuracy_list,predictions_list,labels_list=training.train(num_epochs,train_loader,test_loader,model,error,optimizer,device)

#-----visualizing
visualize.plot(iteration_list,loss_list,accuracy_list)


accuracy.show(test_loader,model,device)

#----print confusion matrix
confusion_mat.show(predictions_list,labels_list,confusion_matrix)