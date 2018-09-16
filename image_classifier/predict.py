#Imports
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
from PIL import Image
import json

#Parser
parser = argparse.ArgumentParser(
    description='This is an Image Predictor',
)

#Arguments
parser.add_argument('image_path', help='Image Folder Path')
parser.add_argument('input', help='Input Checkpoint Path')
parser.add_argument('--top_k', type=int, help='Top K Classes', default=1)
parser.add_argument('--category_names', help='Category Names File Path', default='cat_to_name.json')
parser.add_argument('--gpu', help='Use GPU', action="store_true")
args = parser.parse_args()

#PreProcess Function
def process_image(image):
    #resize
    wpercent = (256/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((256,hsize), Image.ANTIALIAS)
        
    #crop
    width, height = image.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image=image.crop((left, top, right, bottom))
    
    #convert color values
    np_image = np.array(image)
    np_image = np_image/255
    
    #normalization
    np_image = np_image - [0.485, 0.456, 0.406]
    np_image = np_image/[0.229, 0.224, 0.225]
    
    np_image = np_image.transpose((2,0,1))
    
    return np_image

# Predict Function
def predict(image_path, model, topk):
    #Load and pre-process
    im = Image.open(image_path)
    ims = process_image(im)
    imt = torch.FloatTensor(ims).unsqueeze_(0) 
    
    #predict using model
    model.eval()
    with torch.no_grad():
        if args.gpu:
            output = model(imt.to('cuda'))
        else:
            output = model(imt)
    #preparing probabilities
    ps = torch.exp(output.data)
    pst = np.asarray(ps.topk(topk)[0][0])
    #print(pst)
    
    #preparing classes
    index = np.asarray(ps.topk(topk)[1][0])
    inv_map = {v: k for k, v in model.class_to_idx.items()}
    classes = []
    for i in index:
        classes.append(inv_map[i]) 
    #print(classes)
    
    return pst, classes

# Creating function that builds Network
def Network(hidden_layers, arch):
    if arch == 'vgg16':
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


#Load model from checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Network(checkpoint['hidden_layers'],checkpoint['arch'])
    if args.gpu:
        model.to('cuda')
    model.loaded_optimizer = optim.Adam(model.classifier.parameters())                    
    model.loaded_optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epochs']
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    return model

#Load Category Names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

#Main Routine
if __name__ == "__main__":
    probs, classes = predict(args.image_path, load_checkpoint(args.input),args.top_k)

#Printing Results
print("Prediction Result:")
print("")
print ("flower name is: {}".format(cat_to_name[classes[0]]))
print("probability is: {:.3f}".format(probs[0]))
print("")
if args.top_k != 1:
    print("{} most likely cases:".format(args.top_k))
    print("     {}  {}".format('class','category name'))
    for i in classes:
        print("     {}  {}".format(i,cat_to_name[i]))