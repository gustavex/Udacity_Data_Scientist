# This script loads an Image Classifier Model from a checkpoint file.
# The script requires a 
# Train Data is used to train the model.
# Finally the model is saved to a checkpoint file.
#
# Default Settings:
#   -save directory: ./checkpoint.pth
#   -read directory: ./checkpoint.pth
#   -learning rate: 0.001
#   -epochs: 5
#   -hidden layers: 4096 1024
#   -arch: vgg16
#   -gpu: false

#Imports
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

#Parser
parser = argparse.ArgumentParser(
    description='This is an Image Classifier',
)

#Arguments
parser.add_argument('data_directory', help='Data Directory')
parser.add_argument('--save_dir', help='CheckPoint Save Directory and Filename. Defaults to ./checkpoint.pth', default='checkpoint.pth')
parser.add_argument('--read_dir', help='CheckPoint Read Directory and Filename. Defaults to ./checkpoint.pth', default='checkpoint.pth')
parser.add_argument('--learning_rate', type = float, help='Learning Rate. Defaults to 0.001', default=0.001)
parser.add_argument('--epochs', type=int, help='Epoch Number. Defaults to 5', default=5)
parser.add_argument('--hidden_layers', type=int, help='Hidden Layers. Defaults to 4096 1024', nargs=2, default=[4096,1024])
parser.add_argument('--arch', help='Architecture: vgg16(default) or vgg13', default='vgg16', const='vgg16', nargs='?',choices=['vgg16', 'vgg13'])
parser.add_argument('--gpu', help='GPU flag', action="store_true")
args = parser.parse_args()

#Data Directory
data_dir = args.data_directory
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'

#Transform Data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

#Load Datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

#DataLoaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Implement a function for the validation pass
def validation(model, testloader, criterion):
    validation_loss = 0
    accuracy = 0
    for images, labels in testloader:
    # Check gpu flag and calculation validation loss
        if args.gpu:
            output = model.forward(images.to('cuda'))
            validation_loss += criterion(output, labels.to('cuda')).item()
        else:
            output = model.forward(images)
            validation_loss += criterion(output, labels).item()
        # Calculate accuracy
        ps = torch.exp(output)
        if args.gpu:
            equality = (labels.to('cuda').data == ps.max(dim=1)[1])
        else:
            equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validation_loss, accuracy

#Training Model
def train_model(model):
    criterion = nn.NLLLoss()
    # Initial values
    epochs = args.epochs
    print_every = 40
    steps = 0
    # Starting training loop
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            # Check gpu flag
            if args.gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            # Clearing the gradients of the optimizer
            model.loaded_optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            model.loaded_optimizer.step()
            # Calculate running loss
            running_loss += loss.item()
            # Print data loop
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, testloader, criterion)
                # Print data
                print("Epoch: {}/{}.. ".format(model.epochs + e + 1, model.epochs + epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(validation_loss/len(testloader)),
                    "Accuracy: {:.3f}".format(accuracy/len(testloader)))
                # Clearing running loss value
                running_loss = 0
                # Make sure training is back on
                model.train()   
            
# Save the checkpoint and hyperparameters
def save_checkpoint(model):
    model.class_to_idx = train_data.class_to_idx
    model.epochs = model.epochs + args.epochs
    model.arch = args.arch
    checkpoint = {'epochs': model.epochs,
                  'hidden_layers': args.hidden_layers,
                  'state_dict': model.state_dict(),
                  'optimizer': model.loaded_optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'arch':model.arch}
    torch.save(checkpoint, args.save_dir)
    
# Creating function that builds Network
def Network(hidden_layers):
    # Check architecture
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    # Defining classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_layers[0])),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.5)), 
        ('fc3', nn.Linear(hidden_layers[1], 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    # Replacing classifier
    model.classifier = classifier
    return model

# Load the checkpoint and rebuild the model
def load_checkpoint(filepath):
    # Loading checkpoint
    checkpoint = torch.load(filepath)
    # Rebuilding model
    model = Network(args.hidden_layers)
    # Check gpu
    if args.gpu:
        model.to('cuda')
    # Create optimizer
    model.loaded_optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)     
    # Loading model hyperparameters
    model.class_to_idx = checkpoint['class_to_idx']
    model.arch = checkpoint['arch']
    # Check if the provided hidden layers and architecture are the same as in the checkpoint
    # If equal load the saved hyperparameters
    # If different don't load the parameters and set epoch number to zero
    if args.hidden_layers == checkpoint['hidden_layers'] and args.arch == checkpoint['arch'] :
        model.load_state_dict(checkpoint['state_dict'])
        for item in checkpoint['optimizer']['param_groups']:
            item['lr'] = args.learning_rate
        model.loaded_optimizer.load_state_dict(checkpoint['optimizer'])
        model.epochs = checkpoint['epochs']
    else:
        model.epochs = 0
    return model

#Main Routine
if __name__ == "__main__":
    model = load_checkpoint(args.read_dir)
    train_model(model)
    save_checkpoint(model)