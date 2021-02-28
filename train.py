import torch
from abc import ABC, abstractmethod
from .mix import mixup_criterion, MixMethod
from .models import save_checkpoint

class Trainer(ABC): 
    def __init__(self, model, trainloader, criterion, optimizer, lr_scheduler=None,  device="cuda"): 
        self.model = model
        self.trainloader = trainloader
        self.criterion = criterion
        self.optimizer = optimizer 
        self.lr_scheduler = lr_scheduler
        self.device = device
        
    @abstractmethod
    def train(self, epoch): 
        pass  
    
class MixTrainer(Trainer): 
    def __init__(self, model, trainloader, criterion, optimizer, mixmethod, lr_scheduler=None,  device="cuda"): 
        Trainer.__init__(self, model, trainloader, criterion, optimizer, lr_scheduler=lr_scheduler,  device=device)
        self.mixmethod = mixmethod

    def train(self, epoch): 
        print('\nEpoch: %d' % epoch)

        self.model.train()
        self.mixmethod.set_device(self.device)
        train_loss, correct, total = 0, 0, 0 
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            mixed_inputs, targets_a, targets_b, lam = self.mixmethod.mix_data(inputs, targets)
        
            logits = self.model(mixed_inputs)
            
            loss = mixup_criterion(self.criterion, logits, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b).cpu().sum().float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

        train_acc = 100.0*correct/total
        final_loss = train_loss/(batch_idx+1)

        print(f'Loss: {final_loss}| Acc: {train_acc} ({correct}/{total})')
        return final_loss, train_acc
    
class DoubleMixTrainer(Trainer): 
    def __init__(self, model, trainloader, criterion, optimizer, mixmethod, lr_scheduler=None,  device="cuda"): 
        Trainer.__init__(self, model, trainloader, criterion, optimizer, lr_scheduler=lr_scheduler,  device=device)
        self.mixmethod = mixmethod

    def train(self, epoch): 
        print('\nEpoch: %d' % epoch)

        self.model.train()
        self.mixmethod.set_device(self.device)
        train_loss, correct, total = 0, 0, 0 
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            mixed1, mixed2, targets_a, targets_b, lam = self.mixmethod.mix_data(inputs, targets)
        
            logits1 = self.model(mixed1)
            logits2 = self.model(mixed2)
            
            loss1 = mixup_criterion(self.criterion, logits1, targets_a, targets_b, lam)
            loss2 = mixup_criterion(self.criterion, logits2, targets_a, targets_b, lam)
            loss = (loss1 + loss2)*0.5
            
            train_loss += loss.item()
            _, predicted1 = torch.max(logits1, 1)
            _, predicted2 = torch.max(logits2, 1)
            total += targets.size(0)
            
            correct += (lam * predicted1.eq(targets_a).cpu().sum().float()
                        + (1 - lam) * predicted1.eq(targets_b).cpu().sum().float()) * 0.5
            correct += (lam * predicted2.eq(targets_a).cpu().sum().float()
                        + (1 - lam) * predicted2.eq(targets_b).cpu().sum().float()) * 0.5
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

        train_acc = 100.0*correct/total
        final_loss = train_loss/(batch_idx+1)

        print(f'Loss: {final_loss}| Acc: {train_acc} ({correct}/{total})')
        return final_loss, train_acc
    
class Tester(ABC): 
    def __init__(self, model, criterion, testloader, device="cuda"): 
        self.model = model 
        self.criterion = criterion 
        self.testloader  = testloader
        self.device = device 
        
    @abstractmethod
    def test(self, epoch): 
        pass
    
class MixTester(Tester): 
    def test(self, epoch): 
        self.model.eval()
        test_loss, correct, total = 0, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            test_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum()

        test_acc = 100.*correct/total
        final_loss = test_loss/(batch_idx+1)
        
        print(f'Loss: {final_loss}| Acc: {test_acc} ({correct}/{total})')
        return final_loss, test_acc
