import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # → 32x28x28
        self.bn1 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)                           # → 32x14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # → 64x14x14
        self.bn2 = nn.BatchNorm2d(64)                            # → 64x7x7 after pooling

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolution + BatchNorm + ReLU + Pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x