import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D convolutions."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """Residual block with SE unit and optional dilation."""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SpectralCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, embedding_dim=128):
        super(SpectralCNN, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers
        self.layer1 = self._make_layer(64, 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(128, 2, stride=2, dilation=2)
        self.layer3 = self._make_layer(256, 2, stride=2, dilation=4)
        self.layer4 = self._make_layer(512, 2, stride=2, dilation=1)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Heads with improved regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased from 0.3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def _make_layer(self, out_channels, blocks, stride, dilation):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride, dilation))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (Batch, Channels, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, L)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        logits = self.classifier(x)
        embeddings = self.projector(x)
        
        return logits, embeddings

if __name__ == "__main__":
    model = SpectralCNN(num_classes=5)
    dummy_input = torch.randn(8, 1, 2048)
    logits, embs = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embs.shape}")
