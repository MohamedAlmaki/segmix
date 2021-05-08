from enum import Enum
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset 
import PIL.Image as Image
import gdown, tarfile, os, re

def make_dataset(root_dir): 
  expr = "\d+"
  samples = []

  for target in range(0, 10):
    target_dir = os.path.join(root_dir, str(target) + "/")
    indexes = set({})

    for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
      for fname in fnames: 
        indexes.add(int(re.search(expr, fname).group(0)))
    
    for idx in indexes: 
      samples.append((idx, target))

  return samples

class SegmentedCifar10(VisionDataset): 
  """Cifar10 with segmentation dataset."""
  def __init__(self, root_dir, train=True, download=False, transform=None, target_transform=None, attentions_transform=None):
    super(SegmentedCifar10, self).__init__(root_dir, transform=transform, target_transform=target_transform)

    self.root_dir = root_dir
    self.train = train
    self.attentions_transform = attentions_transform 

    if download:
      self.download()
      self.root_dir = os.path.join(root_dir, "cifar10")
    else: 
      if os.path.split(self.root_dir)[-1] != "cifar10": 
        raise Exception("root_dir shoud end with cifar10 folder and this folder should contain train and test folders")

    self.data_dir = os.path.join(self.root_dir, "train" if train else "test")
    self.samples = make_dataset(self.data_dir)

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, index):
    img_idx, target = self.samples[index]
    img, attentions = self.loader(img_idx, target)

    if self.transform is not None: 
      img = self.transform(img)
    
    if self.attentions_transform is not None: 
      attentions = [self.attentions_transform(attn) for attn in attentions]

    if self.target_transform is not None: 
      target = self.target_transform(target)

    return img, attentions, target

  def download(self): 
    url = "https://drive.google.com/u/2/uc?id=1n9BQ1IenIcfa4XU9JMPTFlgRAA3bZDwe&export=download"
    output_path = os.path.join(self.root_dir, 'cifar10-folders.tar.gz')

    gdown.download(url, output_path, quiet=False) 

    print("Extracting....")
    tar = tarfile.open(output_path)
    tar.extractall()
    tar.close()

  def loader(self, index, target):
    target_dir = os.path.join(self.data_dir, str(target) + "/")

    img = Image.open(os.path.join(target_dir, str(index) + ".png"))
    attentions = [Image.open(os.path.join(target_dir, str(index) + "_attn" + str(k) + ".png")) for k in range(0,6)]

    return img, attentions

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