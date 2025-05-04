import torch
import torch.nn as nn
import torch.nn.functional as F

class DistortionGraphModule(nn.Module):
    def __init__(self, input_channels, node_size=(8, 8)):
        """
        Distortion Graph Module as described in the paper.
        
        Args:
            input_channels: Number of input channels
            node_size: Size of the graph (h, w)
        """
        super(DistortionGraphModule, self).__init__()
        self.node_size = node_size
        self.input_channels = input_channels
        
        # Projection layers for node creation
        self.node_projection = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        
        # Graph convolution weights
        self.gc1 = nn.Linear(input_channels, input_channels)
        self.gc2 = nn.Linear(input_channels, input_channels)
        
        # Parameter for residual connection
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        """
        Forward pass through the graph module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Enhanced features
        """
        batch_size = x.size(0)
        
        # Step 1: Node Embedding - Project conventional features to graph space
        # H1 is the set of nodes represented by features in each region
        node_features = self.node_projection(x)  # B x C x H x W
        H1 = node_features.view(batch_size, self.input_channels, -1)  # B x C x (H*W)
        H1 = H1.permute(0, 2, 1)  # B x (H*W) x C
        
        # Step 2: Relationship Modeling - Construct adjacency matrix
        # Normalized dot product similarity matrix
        A = torch.bmm(H1, H1.transpose(1, 2))  # B x (H*W) x (H*W)
        A = F.softmax(A, dim=2)
        
        # Step 3: Graph Reasoning - Two rounds of graph convolution
        H = H1
        H = F.relu(self.gc1(H))  # First graph conv
        H = torch.bmm(A, H)
        
        H = F.relu(self.gc2(H))  # Second graph conv
        H = torch.bmm(A, H)
        
        # Reshape back to original resolution
        H = H.permute(0, 2, 1)  # B x C x (H*W)
        H = H.view(batch_size, self.input_channels, *self.node_size)  # B x C x H x W
        
        # Upsample to match the input resolution
        if H.size(2) != x.size(2) or H.size(3) != x.size(3):
            H = F.interpolate(H, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        
        # Apply residual connection
        output = x + self.alpha * H
        
        return output