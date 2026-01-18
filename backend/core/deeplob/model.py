"""
DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
Implementation of CNN architecture for mid-price movement prediction

The model predicts whether the mid-price will:
- Move UP (class 0)
- Stay STATIONARY (class 1)  
- Move DOWN (class 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class DeepLOBPrediction:
    """Prediction output from DeepLOB"""
    up_probability: float
    stationary_probability: float
    down_probability: float
    predicted_class: int  # 0=UP, 1=STATIONARY, 2=DOWN
    confidence: float
    
    @property
    def direction(self) -> str:
        """Get direction as string"""
        return ["UP", "STATIONARY", "DOWN"][self.predicted_class]


class DeepLOB(nn.Module):
    """
    DeepLOB CNN Architecture
    
    Input: Order book snapshot (40 levels x 4 features x T timesteps)
    Features: [bid_price, bid_volume, ask_price, ask_volume]
    
    Architecture:
    1. Convolutional layers to extract spatial features
    2. Inception modules for multi-scale feature extraction
    3. LSTM for temporal dependencies
    4. Fully connected layers for classification
    """
    
    def __init__(self, 
                 num_levels: int = 40,
                 num_features: int = 4,
                 num_timesteps: int = 100,
                 num_classes: int = 3):
        """
        Args:
            num_levels: Number of order book levels (default 40)
            num_features: Number of features per level (default 4)
            num_timesteps: Number of time steps (default 100)
            num_classes: Number of output classes (default 3)
        """
        super(DeepLOB, self).__init__()
        
        self.num_levels = num_levels
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        
        # Convolutional Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # Convolutional Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        
        # Convolutional Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # Inception Module
        self.inception1 = InceptionModule(32, 64)
        self.inception2 = InceptionModule(64, 64)
        self.inception3 = InceptionModule(64, 64)
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 1, num_levels, num_features * num_timesteps)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Inception modules
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 64)
        
        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep
        
        # Fully connected
        x = self.fc(x)
        
        return x


class InceptionModule(nn.Module):
    """
    Inception module for multi-scale feature extraction
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels // 4)
        )
        
        # 1x1 -> 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels // 4),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels // 4)
        )
        
        # 1x1 -> 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels // 4),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels // 4)
        )
        
        # Max pooling -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels // 4)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all branches"""
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class DeepLOBPredictor:
    """
    Wrapper for DeepLOB model with preprocessing and prediction
    """
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_path: Path to saved model weights
            device: Device to run model on
        """
        self.device = device
        self.model = DeepLOB().to(device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        
    def preprocess_orderbook(self, 
                           orderbook_snapshots: List[Dict]) -> torch.Tensor:
        """
        Preprocess order book snapshots into model input
        
        Args:
            orderbook_snapshots: List of order book snapshots
            
        Returns:
            Tensor of shape (1, 1, num_levels, num_features * num_timesteps)
        """
        num_levels = 40
        num_features = 4
        num_timesteps = len(orderbook_snapshots)
        
        # Initialize array
        data = np.zeros((num_levels, num_features * num_timesteps))
        
        for t, snapshot in enumerate(orderbook_snapshots):
            bids = snapshot.get('bids', [])
            asks = snapshot.get('asks', [])
            
            for i in range(min(num_levels // 2, len(bids))):
                # Bid price and volume
                data[i, t * num_features] = float(bids[i][0])
                data[i, t * num_features + 1] = float(bids[i][1])
            
            for i in range(min(num_levels // 2, len(asks))):
                # Ask price and volume
                data[num_levels // 2 + i, t * num_features + 2] = float(asks[i][0])
                data[num_levels // 2 + i, t * num_features + 3] = float(asks[i][1])
        
        # Normalize
        data = self._normalize(data)
        
        # Convert to tensor
        tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using z-score normalization
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        
        # Avoid division by zero
        std[std == 0] = 1
        
        normalized = (data - mean) / std
        
        return normalized
    
    def predict(self, orderbook_snapshots: List[Dict]) -> DeepLOBPrediction:
        """
        Predict mid-price movement
        
        Args:
            orderbook_snapshots: List of recent order book snapshots
            
        Returns:
            DeepLOBPrediction object
        """
        # Preprocess
        input_tensor = self.preprocess_orderbook(orderbook_snapshots)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        
        # Extract probabilities
        probs = probabilities.cpu().numpy()[0]
        
        up_prob = float(probs[0])
        stationary_prob = float(probs[1])
        down_prob = float(probs[2])
        
        # Predicted class
        predicted_class = int(np.argmax(probs))
        
        # Confidence
        confidence = float(np.max(probs))
        
        return DeepLOBPrediction(
            up_probability=up_prob,
            stationary_probability=stationary_prob,
            down_probability=down_prob,
            predicted_class=predicted_class,
            confidence=confidence
        )
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)


class DeepLOBTrainer:
    """
    Training utilities for DeepLOB
    """
    
    def __init__(self, 
                 model: DeepLOB,
                 learning_rate: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: DeepLOB model
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, 
                test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluate model
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        
        return avg_loss, accuracy
