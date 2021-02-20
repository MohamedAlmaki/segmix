import torch
import numpy as np 
from .models import get_resnet18_modified
from abc import ABC, abstractmethod


class MixMethod(ABC):
    def __init__(self, alpha : float, device: str): 
        self.alpha = alpha
        self.device = device 
         
    @abstractmethod
    def mix_data(self, x, y): 
        pass
    
    def set_device(self, device): 
        self.device = device

class Mixup(MixMethod): 
    def mix_data(self, x, y): 
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
class Segmix(MixMethod): 
    def __init__(self, segmodel): 
        self.segmodel = segmodel
    
    @abstractmethod    
    def mix_data(self, x, y): 
        pass
    
class Cifar10Segmix(Segmix): 
    def attention(self, x):
        return torch.sigmoid(torch.logsumexp(x, 1, keepdim=True))
    
    def get_segmented_images(self, x):
        preds = self.segmodel(x.to(self.device))
        attn = self.attention(preds)
        attn = torch.cat((attn, attn, attn), dim=1)
        attn[attn < 0.3] = 0.0
        attn[attn >= 0.3] = 1.0
        x = x.to(self.device) * attn.to(self.device)
        return x
    
    def mix_data(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]    
        index = torch.randperm(batch_size).to(self.device)

        x2 = self.get_segmented_images(x[index, :])
        
        mixed_x = lam * x.to(self.device) + (1 - lam) * x2.to(self.device)
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
        


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)