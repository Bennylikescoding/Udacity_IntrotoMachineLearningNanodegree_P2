# functions.py
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
from PIL import Image

def load_data(data_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    '''
    input: parent data directory
    output: 6 files, 3 data files transformed by ImageFolder, 3 files transformed by Dataloaders
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Training data transform
    data_transforms_training = transforms.Compose([transforms.RandomRotation(30), # randomly rotate
                                                   transforms.RandomResizedCrop(224), # randomly scale and crop
                                                   transforms.RandomHorizontalFlip(), #randomly flip
                                                   transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], # normalize mean
                                                                      [0.229, 0.224, 0.225])]) # normalize std

    # Validation data transform
    data_transforms_validation = transforms.Compose([transforms.Resize(256), # resize
                                                   transforms.CenterCrop(224), # crop
                                                   transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], # normalize mean
                                                                      [0.229, 0.224, 0.225])]) # normalize std

    # Test data transform
    data_transforms_testing = transforms.Compose([transforms.Resize(256), # resize
                                                   transforms.CenterCrop(224), # crop
                                                   transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], # normalize mean
                                                                      [0.229, 0.224, 0.225])]) # normalize std

    # TODO: Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(train_dir, transform = data_transforms_training)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform = data_transforms_validation)
    image_datasets_test = datasets.ImageFolder(test_dir, transform = data_transforms_testing)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # Shuffle here is extremely important. If we don't add, the validation accuracy would be very low around 0.01
    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size = 50, shuffle = True) 
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size = 50)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size = 50)

    return image_datasets_train,image_datasets_valid,image_datasets_test,dataloaders_train,dataloaders_valid,dataloaders_test

def build_classifier(model, input_units, hidden_units, class_numbers, dropout):
    '''
    input: model name, numbers of hidden units, dropout ratio
    output: new combined model
    '''
    # Freeze parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    new_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units, bias=True)), # 25088 and 4096 is the original in_ and out_features of vgg network
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p = dropout)),
        ('fc2', nn.Linear(hidden_units, class_numbers, bias=True)), 
        ('output', nn.LogSoftmax(dim=1))
        ]))

    if model == 'resnet' or model == 'inception':

        model.fc = new_classifier # replace original classifier with the new classifier
        return model

    else: # for models of alexnet, vgg, squeezenet, densenet:

        model.classifier = new_classifier # replace original classifier with the new classifier
        return model

def train_and_validation_the_model(gpu_mode,model,learning_rate,epochs,dataloaders_train,dataloaders_valid,criterion,optimizer):
    '''
    input: model,
    output: 
    '''
    if gpu_mode == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        pass

    # define loss and optimizer
    criterion = nn.NLLLoss()

    # Note that we here only train the classifier because the feature layers are frozen
    if model == 'resnet' or model == 'inception':

        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    else: # for models of alexnet, vgg, squeezenet, densenet:

        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
 
    # Now start training
    epochs = epochs
    steps = 0
    print_every = 30

    for epoch in range(epochs):

        model.train()
        running_loss = 0
            
        for images, labels in dataloaders_train:
            steps += 1
            # Move input and label tensors to the default device
            # Use GPU if it's available
            device = device

            model.to(device)

            images, labels = images.to(device), labels.to(device)
            
            # now start training pass
            optimizer.zero_grad()
            
            # forward and backward loop
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:

                model.eval()
                
                with torch.no_grad():
                    Val_loss = 0
                    accuracy = 0               

                    for images2, labels2 in dataloaders_valid:

                        images2, labels2 = images2.to(device), labels2.to(device)
                        output2 = model.forward(images2)
                        #batch_loss = criterion(output, labels)
                        Val_loss += criterion(output2, labels2).item()
                        #Val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(output2)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels2.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Val loss: {Val_loss/len(dataloaders_valid):.3f}.. "
                      f"Val accuracy: {accuracy/len(dataloaders_valid):.3f}"
                     "\n, training in progress...")
                
                running_loss = 0
                model.train()

    return model

def test_model(model, dataloaders_test, gpu_mode):
    # TODO: Do validation on the test set
    correct_prediction = 0
    total_num_test_images = 0

    if gpu_mode == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        pass

    with torch.no_grad():
        model.eval()
        for images, labels in dataloaders_test:
            device = device
            
            model.to(device)
            images, labels = images.to(device), labels.to(device)
            # Return predicted labels
            predicted_prob = model.forward(images)
            # Turn max probabilities into corresponded labels
            _, predicted_labels = torch.max(predicted_prob.data, 1)
            # check numbers of correct prediction
            total_num_test_images += labels.size(0)
            correct_prediction += (predicted_labels == labels).sum().item()

    print (f"test prediction pbs: {correct_prediction / total_num_test_images:.3f}") 

def save_checkpoint(model_name, dataset_after_ImageFolder,chekpoint_name,arch):
    model_name.class_to_idx = dataset_after_ImageFolder.class_to_idx
    checkpoint = {'model info': model_name,
                 'classifier': model_name.classifier,
                 'class_to_idx': model_name.class_to_idx,
                 'state_dict': model_name.state_dict(),
                 'model structure': arch}

    torch.save(checkpoint, chekpoint_name)
    print (chekpoint_name, "checkpoint has saved !")
    return checkpoint

def load_checkpoint(pretrained_model, filepath):
    checkpoint = torch.load(filepath)
    
    pretrained_model.class_to_idx = checkpoint['class_to_idx']
    pretrained_model.classifier = checkpoint['classifier']
    pretrained_model.load_state_dict(checkpoint['state_dict'])
    
    # freeze parameters
    for param in pretrained_model.parameters(): 
        param.requires_grad = False
    
    return pretrained_model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # load image
    im = Image.open(image)

    # get width and height of original image
    ori_w, ori_h = im.size
    print ("ori w&h", ori_w, ori_h)
    
    # resize image according to conditions
    if ori_w <= ori_h:
        # use 'thumbnail' instead of 'resize' to keep the original ratio
        im.thumbnail((256, ori_h))
    else:
        im.thumbnail((ori_w, 256))

    # get new width and height of image
    new_w, new_h = im.size
    print ("new w&h", new_w, new_h)
    
    # set coordinates for l, t, r, b and crop the image
    required_dimension = 224, 224 # w, h = 224, 224
    
    #the coordinate system for crop: cropped_img = img.crop( ( x, y, x + width , y + height ) )
    left = (new_w - required_dimension[0]) / 2
    top = (new_h - required_dimension[1]) / 2
    right = (left + required_dimension[0])
    bottom = (top + required_dimension[1])

    im_resized_cropped = im.crop((left, top, right, bottom))

    # convert image to numpy and standardize by dividing 255
    np_im = np.array(im_resized_cropped) / 255

    # normalize using (x - u)/o on all 3 RGB channels
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    np_im = (np_im - normalize_mean) / normalize_std

    # set color to the first, the original is 0, 1, 2
    np_im = np_im.transpose(2, 0, 1)

    return np_im

def predicting(gpu_mode, image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    if gpu_mode == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        model.to('cpu')
        
    model.eval();

    # use 'process_image()', transfer raw image -> model input (numpy array)
    model_input = process_image(image_path)

    # convert numpy array to torch
    t_im = torch.from_numpy(model_input).type(torch.FloatTensor)

    # add dimension
    t_im_dim = t_im.unsqueeze(dim=0)

    # now start predicting
    with torch.no_grad():
        if gpu_mode == True:
            output = model.forward(t_im_dim.cuda())
        else:
            output=model.forward(t_im_dim)
        
        ps = torch.exp(output)
        top_p, top_index = ps.topk(topk, dim=1)

        #make top_p and top_index as list
        top_p_list = np.array(top_p[0])
        top_index_list = np.array(top_index[0])

        #load index
        class_to_idx = model.class_to_idx

        #invert the dictionary so you get a mapping from index to class as well.
        index_class_dict = {x: y for y, x in class_to_idx.items()}

        # get class list
        class_top_lst = []
        for i in top_index_list:
            class_top_lst += [index_class_dict[i]]
            
        return top_p_list, class_top_lst 







