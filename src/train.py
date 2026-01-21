import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
from src.model import SpectralCNN

class SpectralDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample.unsqueeze(0), label # Add channel dimension

def spectral_augmentation(x, noise_level=0.01, shift_range=5):
    """Basic augmentation for spectral data."""
    # 1. Add Gaussian noise
    noise = torch.randn_like(x) * noise_level
    x = x + noise
    
    # 2. Random shift (circular)
    shift = np.random.randint(-shift_range, shift_range + 1)
    x = torch.roll(x, shifts=shift, dims=-1)
    
    # 3. Random scaling
    scale = 1.0 + (np.random.rand() - 0.5) * 0.1
    x = x * scale
    
    return x

class Trainer:
    def __init__(self, model, device='cpu', log_dir='logs'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.writer = SummaryWriter(os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))
        print(f"Trainer initialized on device: {device}")
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", unit="batch", leave=False)
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': f"{running_loss/(i+1):.4f}"})
            
        acc = 100. * correct / total
        avg_loss = running_loss / len(dataloader)
        
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/train', acc, epoch)
        
        return avg_loss, acc

    def validate(self, dataloader, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", unit="batch", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits, _ = self.model(inputs)
                loss = self.criterion(logits, targets)
                
                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        acc = 100. * correct / total
        avg_loss = running_loss / len(dataloader)
        
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/val', acc, epoch)
        
        return avg_loss, acc

    def save_checkpoint(self, path):
        torch.save(self.model.state_state_dict(), path)
        print(f"Checkpoint saved to {path}")

if __name__ == "__main__":
    # Sanity check with dummy data
    X_dummy = np.random.rand(100, 2048)
    y_dummy = np.random.randint(0, 5, 100)
    
    dataset = SpectralDataset(X_dummy, y_dummy, transform=spectral_augmentation)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = SpectralCNN(num_classes=5)
    trainer = Trainer(model)
    
    print("Starting dummy training sanity check...")
    for epoch in range(2):
        loss, acc = trainer.train_epoch(loader, epoch)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.2f}%")
