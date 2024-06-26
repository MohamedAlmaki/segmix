import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import os
from enum import Enum

class Model(Enum): 
    resnet18 = "resnet18"
    resnet18_modified = "resnet18_modified"
    resnet50 = "resnet50"
    
class ModelsException(Exception): 
    pass 

def get_resnet50(pretrained=False, num_classes=10):
    model = resnet50(pretrained=pretrained)
    model.fc = nn.Linear(2048, num_classes)
    return model

def get_resnet18(pretrained=False, num_classes=10):
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model

def get_resnet18_modified(pretrained=False, num_classes=10): 
    resnet = resnet18(pretrained=pretrained)
    resnet.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
    resnet_ = list(resnet.children())[:-2]
    resnet_[3] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    classifier = nn.Conv2d(512, num_classes,1)
    torch.nn.init.kaiming_normal_(classifier.weight)
    resnet_.append(classifier)
    resnet_.append(nn.Upsample(size=32, mode='bilinear', align_corners=False))
    tiny_resnet = nn.Sequential(*resnet_)
    model = nn.DataParallel(tiny_resnet)
    return model

def load_model(name, pretrained=False, num_classes=10): 
    if name == Model.resnet18: 
        return get_resnet18(pretrained=pretrained, num_classes=num_classes)
    elif name == Model.resnet18_modified: 
        return get_resnet18_modified(pretrained=pretrained, num_classes=num_classes)
    elif name == Model.resnet50: 
        return get_resnet50(pretrained=pretrained, num_classes=num_classes)
    else: 
        raise ModelsException("Unknown model name. please select one of the supported models")
    
def save_checkpoint(model, path, optimizer, epoch, best_acc): 
    state = {
        "state": model.state_dict(),
        "optim": optimizer.state_dict(), 
        "epoch": epoch, 
        "best_acc": best_acc
    }
    torch.save(state, path) 
    
def load_checkpoint(path, model, optimizer):
    assert(os.path.exists(path))
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'])
    optimizer.load_state_dict(checkpoint['optim'])
    last_epoch = int(checkpoint['epoch'])
    best_acc = float(checkpoint['best_acc'])
    
    return model, optimizer, last_epoch, best_acc
    
    
        
