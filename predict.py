#predict.py
#import libraries
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from collections import OrderedDict
import argparse
import json

from functions import load_checkpoint,process_image,predicting

# add argument definition
# ref: https://pymotw.com/3/argparse/
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',help='directory for model checkpoint')


parser.add_argument('--test_image_path', type=str, default='flowers/test/11/image_03141.jpg', dest='test_image_path', help='path for test image, string format')

parser.add_argument('--topk', dest='topk',type=int, default=5, help='enter top k value of class to view')

parser.add_argument('--cat_to_name', dest='cat_to_name',default='cat_to_name.json', help='load cat_to_name.json file')

parser.add_argument('--gpu', dest='gpu',action='store_true', help='set gpu model, default is gpu')

results = parser.parse_args()

# from the parser, get variables
checkpoint_dir = results.checkpoint_dir
test_image_path = results.test_image_path
topk = results.topk
cat_to_name = results.cat_to_name
gpu_mode = results.gpu

with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f)

# load model backbone
checkpoint = torch.load(checkpoint_dir)
load_arch = checkpoint['model structure']
model = getattr(models,load_arch)(pretrained=True)

# load model checkpoint
loaded_model = load_checkpoint(model, checkpoint_dir)

# Image processing (included in the 'predict' function) and predict the top classification
probs,classes = predicting(gpu_mode, test_image_path, loaded_model, topk=topk)

print(probs)
print(classes)

names = []
index = 0
for i in classes:
    names += [cat_to_name[i]]
    print("This flower is likely to be: ", names[index], ", with a probability of ", probs[index])
    index += 1

print("\nThis flower is most likely to be: ", names[0], ". Probability is ", probs[0])



