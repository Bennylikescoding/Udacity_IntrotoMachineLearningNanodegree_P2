#train.py
# import libraries
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import json
import torchvision.models as models
from collections import OrderedDict
import argparse

from functions import load_data,build_classifier,train_and_validation_the_model,test_model,save_checkpoint

# add argument definition
# ref: https://pymotw.com/3/argparse/
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, dest='data_dir', help='directory for training dataset, string format')

parser.add_argument('--save_dir', type=str, dest='save_dir',help='directory for model checkpoint saving, string format')

parser.add_argument('--arch', type=str, dest='arch',help='choose one architecture from torchvision.models, input is a string, default = VGG16')

parser.add_argument('--learning_rate', type=float, default=0.001, dest='learning_rate',help='define learning rate, default = 0.001')

parser.add_argument('--epochs', type=int, dest='epochs',help='number of epochs trained, default = 4')

parser.add_argument('--dropout', type=float, dest='dropout',help='dropout rate for training, default = 0.2')

parser.add_argument('--input_unit', type=int, dest='input_unit', default=25088, help='define input_unit number, default is 25088 (vgg16)')

parser.add_argument('--hidden_unit', type=int, dest='hidden_unit', default=4096, help='define hidden_unit number, default is 4096 (vgg16)')

parser.add_argument('--class_number', type=int, dest='class_number', default=102, help='define the output class number, default is 102 (for flowers classification project)')

parser.add_argument('--structure', type=str, dest='structure', default='vgg16', help='define the model architecture, default is vgg16')

parser.add_argument('--gpu', dest='gpu',action='store_true', help='set gpu model, default is gpu')


results = parser.parse_args()

# from the parser, get variables
data_dir = results.data_dir
chekpoint_name = results.save_dir
arch = results.arch
learning_rate = results.learning_rate
epochs = results.epochs
dropout = results.dropout
input_unit = results.input_unit
hidden_unit = results.hidden_unit
class_number = results.class_number
structure = results.structure
gpu_mode = results.gpu

# load and preprocess data
image_datasets_train,image_datasets_valid,image_datasets_test,dataloaders_train,dataloaders_valid,dataloaders_test = load_data(data_dir)

# load pretrained model and add new classifiers
model = getattr(models,arch)(pretrained=True)
build_classifier(model, input_unit, hidden_unit, class_number, dropout)

# set criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# model training and validation
print('prerequisite has been fulfiled, now start training & validation...')
model = train_and_validation_the_model(gpu_mode,model,learning_rate,epochs,dataloaders_train,dataloaders_valid,criterion,optimizer)

# model testing
print ('\n now start testing...')
test_model(model, dataloaders_test, gpu_mode)

# model saving
print ('\n now start saving...')
checkpoint = save_checkpoint(model,image_datasets_train,chekpoint_name,arch)








