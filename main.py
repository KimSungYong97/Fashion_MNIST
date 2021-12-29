import pandas as pd

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

train_set = data.FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_set = data.FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,), std=(0.5,))])) # Add Normalization
# array형 이미지이기 때문에 Data Augmentation 은 불가

train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(test_set, batch_size=100)


model = CNN.FashionCNN()
model.to(device)

error = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # can add (weight_decay=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma= 0.999) #Add Scheduler
print(model)

#------Training model & Testing dataset------


num_epochs = 50
loss_list = []
iteration_list = []
accuracy_list = []
predictions_list = []
labels_list = []

loss_list,iteration_list,accuracy_list,predictions_list,labels_list=training.train(num_epochs,train_loader,test_loader,model,error,optimizer,scheduler,device)

#-----Visualizing result
visualize.plot(iteration_list,loss_list,accuracy_list)


accuracy.show(test_loader,model,device)

#----print confusion matrix
confusion_mat.show(predictions_list,labels_list,confusion_matrix)