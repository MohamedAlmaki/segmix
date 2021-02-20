import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np 
import csv
from enum import Enum
from models import Model, load_model, save_checkpoint, load_checkpoint
from datasets import Dataset, getTransfroms, getDatasetLoaders
from train import MixTrainer, MixTester
from mix import Mixup
from logger import CSVLogger

def main(): 
    device = "cpu"  
    resume = True
    mpath = "./checkpoint.pth"
    logpath = "./log.csv"
    dpath = "./dataset"
    
    model = load_model(Model.resnet18)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,78,eta_min=0.001)
    mixmethod = Mixup(0.3, device)
    epochs = 30
    best_acc, s_epoch = 0, 0
    
    if resume: 
        model, optimizer, s_epoch, best_acc  = load_checkpoint(mpath, model, optimizer)

    train_transforms, test_transforms = getTransfroms()
    trainset, trainloader, testset, testloader, classes = getDatasetLoaders(Dataset.cifar10, dpath, train_transforms, test_transforms)
    
    trainer = MixTrainer(model, trainloader, criterion, optimizer, mixmethod, lr_scheduler=lr_scheduler, device=device)
    tester = MixTester(model, criterion, testloader, device=device)
    logger = CSVLogger(logpath, ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])
    
    for epoch in range(s_epoch +  1, epochs):
        train_loss, train_acc = trainer.train(epoch)
        test_loss, test_acc = tester.test(epoch)
        
        if(test_acc > best_acc): 
            best_acc = test_acc
            save_checkpoint(model, mpath, optimizer, epoch, best_acc)
            
        logger.log([epoch, train_loss, train_acc, test_loss, test_acc])
            
if __name__ == "__main__": 
    main()