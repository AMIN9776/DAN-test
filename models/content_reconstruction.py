import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DeformableConv2d, self).__init__()
        
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        # Get offsets
        offsets = self.offset_conv(x)
        
        # Apply deformable convolution
        x = deform_conv2d(x, offsets, self.conv.weight, self.conv.bias, 
                          stride=self.conv.stride, padding=self.conv.padding)
        return x

class SameResolutionBlock(nn.Module):
    def __init__(self, channels):
        super(SameResolutionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ContentReconstructionNetwork(nn.Module):
    def __init__(self, in_channels=3):
        """
        Content Reconstruction Network with Deformable Convolutions.
        
        Args:
            in_channels: Number of input image channels
        """
        super(ContentReconstructionNetwork, self).__init__()
        
        # Initial deformable convolution for feature extraction
        self.deform_conv = DeformableConv2d(in_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Same-resolution feature block
        self.same_res_block = SameResolutionBlock(64)
        
        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )
        
        # Decoder blocks
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction layer
        self.final = nn.Conv2d(64, in_channels, kernel_size=1)
        
    def grid_sample_features(self, features, flow):
        """
        Apply flow-based warping to features using grid_sample.
        
        Args:
            features: Feature maps to warp
            flow: Flow map for warping
            
        Returns:
            Warped features
        """
        B, _, H, W = features.size()
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid_y = grid_y.float().to(flow.device)
        grid_x = grid_x.float().to(flow.device)
        
        # Apply flow to grid
        grid_x = grid_x + flow[:, 0]
        grid_y = grid_y + flow[:, 1]
        
        # Normalize grid coordinates to [-1, 1]
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        # Stack and reshape for grid_sample
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Apply sampling
        warped_features = F.grid_sample(features, grid, mode='bilinear', align_corners=True)
        
        return warped_features
        
    def forward(self, x, flows):
        """
        Forward pass of the content reconstruction network.
        
        Args:
            x: Input fisheye image
            flows: Dictionary of flow maps at different resolutions
            
        Returns:
            Reconstructed undistorted image
        """
        # Initial feature extraction with deformable convolution
        x_deform = F.relu(self.bn1(self.deform_conv(x)))
        
        # Same-resolution feature block
        same_res_features = self.same_res_block(x_deform)
        
        # Encoder with flow-based warping
        e1 = self.enc1(same_res_features)  # 256x256
        e1_warped = self.grid_sample_features(e1, flows['flow256'])
        
        e2 = self.enc2(e1)  # 128x128
        e2_warped = self.grid_sample_features(e2, flows['flow128'])
        
        e3 = self.enc3(e2)  # 64x64
        e3_warped = self.grid_sample_features(e3, flows['flow64'])
        
        e4 = self.enc4(e3)  # 32x32
        e4_warped = self.grid_sample_features(e4, flows['flow32'])
        
        # Bottleneck
        b = self.bottleneck(e4)  # 16x16
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([b, e4_warped], dim=1))
        d3 = self.dec3(torch.cat([d4, e3_warped], dim=1))
        d2 = self.dec2(torch.cat([d3, e2_warped], dim=1))
        d1 = self.dec1(torch.cat([d2, e1_warped], dim=1))
        
        # Final output
        out = self.final(d1)
        
        return out