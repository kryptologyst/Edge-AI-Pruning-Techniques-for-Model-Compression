"""Neural network model architectures."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class MNISTCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize MNIST CNN.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        
        self.input_shape = config.input_shape
        self.num_classes = config.num_classes
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def _get_flattened_size(self) -> int:
        """Calculate flattened size after conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MobileNetV2Tiny(nn.Module):
    """Tiny MobileNetV2 for edge deployment."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize MobileNetV2 Tiny.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        
        self.num_classes = config.num_classes
        self.input_shape = config.input_shape
        
        # Inverted residual blocks
        self.features = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            
            # Inverted residual blocks
            self._make_layer(16, 24, 2, 1),
            self._make_layer(24, 32, 2, 2),
            self._make_layer(32, 64, 2, 2),
            self._make_layer(64, 96, 1, 1),
            self._make_layer(96, 160, 2, 1),
            self._make_layer(160, 320, 1, 1),
            
            # Final conv layer
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, self.num_classes),
        )
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> nn.Sequential:
        """Create inverted residual block.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            stride: Stride for depthwise conv.
            expand_ratio: Expansion ratio.
            
        Returns:
            Inverted residual block.
        """
        layers = []
        
        # Expansion layer
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, in_channels * expand_ratio, 1, bias=False),
                nn.BatchNorm2d(in_channels * expand_ratio),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                in_channels * expand_ratio,
                in_channels * expand_ratio,
                3,
                stride,
                1,
                groups=in_channels * expand_ratio,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=True),
        ])
        
        # Projection layer
        layers.extend([
            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class EfficientNetTiny(nn.Module):
    """Tiny EfficientNet for edge deployment."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize EfficientNet Tiny.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        
        self.num_classes = config.num_classes
        self.input_shape = config.input_shape
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            self._make_mbconv(32, 16, 1, 1, 1),
            self._make_mbconv(16, 24, 2, 6, 1),
            self._make_mbconv(24, 40, 2, 6, 1),
            self._make_mbconv(40, 80, 2, 6, 1),
            self._make_mbconv(80, 112, 1, 6, 1),
            self._make_mbconv(112, 192, 2, 6, 1),
            self._make_mbconv(192, 320, 1, 6, 1),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, self.num_classes),
        )
    
    def _make_mbconv(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        num_layers: int,
    ) -> nn.Sequential:
        """Create MBConv block.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            stride: Stride for depthwise conv.
            expand_ratio: Expansion ratio.
            num_layers: Number of layers in block.
            
        Returns:
            MBConv block.
        """
        layers = []
        
        for i in range(num_layers):
            layers.append(
                MBConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio,
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        """Initialize MBConv block.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            stride: Stride for depthwise conv.
            expand_ratio: Expansion ratio.
        """
        super().__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expand_ratio
        
        # Expansion layer
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.expand = None
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                3,
                stride,
                1,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
        )
        
        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        residual = x
        
        if self.expand is not None:
            x = self.expand(x)
        
        x = self.depthwise(x)
        x = self.project(x)
        
        if self.use_residual:
            x = x + residual
        
        return x
