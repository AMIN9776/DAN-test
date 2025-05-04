import torch
import torch.nn as nn
from models.unwrapping_flow_generator import UnwrappingFlowGenerator
from models.content_reconstruction import ContentReconstructionNetwork

class DAN(nn.Module):
    def __init__(self, in_channels=3):
        """
        Distortion-aware Network (DAN) for fisheye image rectification.
        
        Args:
            in_channels: Number of input image channels
        """
        super(DAN, self).__init__()
        
        # Unwrapping flow generation network
        self.flow_generator = UnwrappingFlowGenerator(in_channels)
        
        # Content reconstruction network
        self.reconstruction_network = ContentReconstructionNetwork(in_channels)
        
    def forward(self, x):
        """
        Forward pass of the DAN model.
        
        Args:
            x: Input fisheye image
            
        Returns:
            Dictionary containing:
                - 'rectified': Rectified image
                - 'flows': Flow maps at different resolutions
        """
        # Generate unwrapping flows
        flows = self.flow_generator(x)
        
        # Rectify the image using content reconstruction network
        rectified = self.reconstruction_network(x, flows)
        
        return {
            'rectified': rectified,
            'flows': flows
        }