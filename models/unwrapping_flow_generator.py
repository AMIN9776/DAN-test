import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_module import DistortionGraphModule

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UnwrappingFlowGenerator(nn.Module):
    def __init__(self, in_channels=3):
        """
        Unwrapping Flow Generation Network with Graph Reasoning.
        
        Args:
            in_channels: Number of input image channels
        """
        super(UnwrappingFlowGenerator, self).__init__()
        
        # Encoder blocks
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 512)
        
        # Distortion graph module at resolution 8x8
        self.graph_module = DistortionGraphModule(512, node_size=(8, 8))
        
        # Decoder blocks
        self.dec5 = ConvBlock(512 + 512, 512)  # +512 for skip connection
        self.dec4 = ConvBlock(512 + 256, 256)  # +256 for skip connection
        self.dec3 = ConvBlock(256 + 128, 128)  # +128 for skip connection
        self.dec2 = ConvBlock(128 + 64, 64)    # +64 for skip connection
        self.dec1 = ConvBlock(64, 32)
        
        # Flow prediction heads at multiple scales
        self.flow_pred8 = nn.Conv2d(512, 2, kernel_size=3, padding=1)
        self.flow_pred16 = nn.Conv2d(512, 2, kernel_size=3, padding=1)
        self.flow_pred32 = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.flow_pred64 = nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.flow_pred128 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.flow_pred256 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        """
        Forward pass of the unwrapping flow generator.
        
        Args:
            x: Input image of shape (B, C, H, W)
            
        Returns:
            Dictionary of flow maps at different resolutions
        """
        # Encoder
        e1 = self.enc1(x)  # 256x256
        e2 = self.enc2(self.pool(e1))  # 128x128
        e3 = self.enc3(self.pool(e2))  # 64x64
        e4 = self.enc4(self.pool(e3))  # 32x32
        e5 = self.enc5(self.pool(e4))  # 16x16
        
        # Bottleneck with graph reasoning
        bottleneck = self.pool(e5)  # 8x8
        bottleneck = self.graph_module(bottleneck)  # Enhanced with graph reasoning
        
        # Flow at 8x8 resolution
        flow8 = self.flow_pred8(bottleneck)
        
        # Decoder with skip connections
        d5 = self.dec5(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), e5], dim=1))
        flow16 = self.flow_pred16(d5)
        
        d4 = self.dec4(torch.cat([F.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=True), e4], dim=1))
        flow32 = self.flow_pred32(d4)
        
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        flow64 = self.flow_pred64(d3)
        
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        flow128 = self.flow_pred128(d2)
        
        d1 = self.dec1(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True))
        flow256 = self.flow_pred256(d1)
        
        # Return flow maps at multiple scales
        flows = {
            'flow8': flow8,
            'flow16': flow16,
            'flow32': flow32,
            'flow64': flow64,
            'flow128': flow128,
            'flow256': flow256
        }
        
        return flows