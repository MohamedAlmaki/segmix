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
    
class CutMix(MixMethod): 
  def rand_bbox(self, size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

  def mix_data(self, x, y): 
    if self.alpha > 0:
      lam = np.random.beta(self.alpha, self.alpha)
    else:
      lam = 1

    rand_index = torch.randperm(x.size()[0]).to(self.device)

    target_a = y
    target_b = y[rand_index]

    bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, target_a, target_b, lam 

class Segmix(MixMethod): 
    def __init__(self, segmodel, alpha, device): 
        super().__init__(alpha, device)
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