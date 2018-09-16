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
parser.add_argument('--save_dir', help='CheckPoint Save Directory/Filename', default='checkpoint.pth')
parser.add_argument('--read_dir', help='CheckPoint Read Directory/Filename', default='checkpoint.pth')
parser.add_argument('--learning_rate', type = float, help='Learning Rate', default=0.001)
parser.add_argument('--epochs', type=int, help='Epochs', default=5)
parser.add_argument('--hidden_layers', type=int, help='Hidden Layers', nargs=2, default=[4096,1024])
parser.add_argument('--arch', help='Architecture: vgg16(default) or vgg13', default='vgg16', const='vgg16', nargs='?',choices=['vgg16', 'vgg13'])
parser.add_argument('--gpu', help='Use GPU', action="store_true")
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
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        if args.gpu:
            output = model.forward(images.to('cuda'))
            test_loss += criterion(output, labels.to('cuda')).item()
        else:
            output = model.forward(images)
            test_loss += criterion(output, labels).item()
            
        ps = torch.exp(output)
        if args.gpu:
            equality = (labels.to('cuda').data == ps.max(dim=1)[1])
        else:
            equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

#Training Model
def train_model(model):
    criterion = nn.NLLLoss()
    #optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    #optimizer = model.loaded_optimizer
    
    epochs = args.epochs
    print_every = 40
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            if args.gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            model.loaded_optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            model.loaded_optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
             
                print("Epoch: {}/{}.. ".format(model.epochs + e + 1, model.epochs + epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                #print(model.loaded_optimizer)
                running_loss = 0
            
                # Make sure training is back on
                model.train()   
            
# Save the checkpoint 
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
    
    model.classifier = classifier
    return model

# Load the checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Network(args.hidden_layers)
    if args.gpu:
        model.to('cuda')
    model.loaded_optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)     
    model.class_to_idx = checkpoint['class_to_idx']
    model.arch = checkpoint['arch']
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
model = load_checkpoint(args.read_dir)
train_model(model)
save_checkpoint(model)