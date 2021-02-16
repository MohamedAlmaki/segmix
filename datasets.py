from enum import Enum
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class Dataset(Enum): 
    cifar10 = "cifar10"
    
def getTransfroms(): 
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return train_transforms, test_transforms
    
def getDatasetLoaders(name, path, train_transforms, test_transforms): 
    trainset, trainloader, testset, testloader, classes  = None, None, None, None, None 
    if name == Dataset.cifar10: 
        trainset = CIFAR10(root=path, train=True, download=True, transform=train_transforms)
        trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

        testset = CIFAR10(root=path, train=False, download=True, transform=test_transforms)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=16, pin_memory=True)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes