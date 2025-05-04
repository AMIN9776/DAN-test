import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultiScaleLoss(nn.Module):
    def __init__(self, decay=0.9, num_scales=5):
        """
        Multi-scale loss based on weight decay as described in the paper.
        
        Args:
            decay: Decay factor for weighting different scales
            num_scales: Number of scales to consider
        """
        super(MultiScaleLoss, self).__init__()
        self.decay = decay
        self.num_scales = num_scales
        
    def forward(self, pred_img, target_img):
        """
        Compute multi-scale loss.
        
        Args:
            pred_img: Predicted image
            target_img: Ground truth image
            
        Returns:
            Multi-scale loss value
        """
        loss = 0.0
        
        for i in range(self.num_scales):
            scale_factor = 1 / (2 ** i)
            
            # Scale the images
            if scale_factor < 1.0:
                pred_scaled = F.interpolate(pred_img, scale_factor=scale_factor, 
                                           mode='bilinear', align_corners=True)
                target_scaled = F.interpolate(target_img, scale_factor=scale_factor, 
                                             mode='bilinear', align_corners=True)
            else:
                pred_scaled = pred_img
                target_scaled = target_img
            
            # Compute L1 loss at this scale with decay weight
            scale_weight = self.decay ** (self.num_scales - 1 - i)
            scale_loss = torch.mean(torch.abs(pred_scaled - target_scaled)) * scale_weight
            
            loss += scale_loss
            
        return loss

class ContentStyleLoss(nn.Module):
    def __init__(self, content_weight=1.0, style_weight=1000.0):
        """
        Content and style loss using VGG features.
        
        Args:
            content_weight: Weight for content loss
            style_weight: Weight for style loss
        """
        super(ContentStyleLoss, self).__init__()
        
        # Load pre-trained VGG16 model
        vgg = models.vgg16(pretrained=True).features.eval()
        self.model = vgg
        
        # Define content and style layers
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        
        # Layer mapping
        self.layer_mapping = {
            'conv1_1': '1',
            'conv2_1': '6',
            'conv3_1': '11',
            'conv4_1': '18',
            'conv4_2': '20',
            'conv5_1': '25'
        }
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Freeze VGG parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def gram_matrix(self, input):
        """Compute Gram matrix for style loss."""
        batch_size, c, h, w = input.size()
        features = input.view(batch_size, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def forward(self, pred_img, target_img):
        """
        Compute content and style loss.
        
        Args:
            pred_img: Predicted image
            target_img: Ground truth image
            
        Returns:
            Content and style loss
        """
        content_loss = 0.0
        style_loss = 0.0
        
        # Normalize images for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred_img.device)
        
        pred_img_norm = (pred_img - mean) / std
        target_img_norm = (target_img - mean) / std
        
        # Extract features
        pred_features = {}
        target_features = {}
        
        x_pred = pred_img_norm
        x_target = target_img_norm
        
        for name, layer in self.model._modules.items():
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            # Save content features
            if name in [self.layer_mapping[layer_name] for layer_name in self.content_layers]:
                pred_features[name] = x_pred
                target_features[name] = x_target
                
                # Compute content loss
                content_loss += F.mse_loss(x_pred, x_target)
            
            # Save style features
            if name in [self.layer_mapping[layer_name] for layer_name in self.style_layers]:
                # Compute style loss
                gram_pred = self.gram_matrix(x_pred)
                gram_target = self.gram_matrix(x_target)
                style_loss += F.mse_loss(gram_pred, gram_target)
        
        # Combine losses
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        return total_loss

class DANLoss(nn.Module):
    def __init__(self, multi_scale_decay=0.9, num_scales=5, 
                 content_weight=1.0, style_weight=1000.0):
        """
        Combined loss function for DAN model.
        
        Args:
            multi_scale_decay: Decay factor for multi-scale loss
            num_scales: Number of scales for multi-scale loss
            content_weight: Weight for content loss
            style_weight: Weight for style loss
        """
        super(DANLoss, self).__init__()
        
        self.l1_loss = nn.L1Loss()
        self.multi_scale_loss = MultiScaleLoss(decay=multi_scale_decay, num_scales=num_scales)
        self.content_style_loss = ContentStyleLoss(content_weight, style_weight)
        
    def forward(self, outputs, target, flow_gt=None):
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dictionary
            target: Ground truth image
            flow_gt: Ground truth flow (optional)
            
        Returns:
            Combined loss value and individual loss components
        """
        # Basic L1 loss on rectified image
        l1_loss = self.l1_loss(outputs['rectified'], target)
        
        # Multi-scale loss
        ms_loss = self.multi_scale_loss(outputs['rectified'], target)
        
        # Content-style loss
        cs_loss = self.content_style_loss(outputs['rectified'], target)
        
        # Flow loss if ground truth flow is provided
        flow_loss = 0.0
        if flow_gt is not None:
            pred_flow = outputs['flows']['flow256']
            flow_loss = self.l1_loss(pred_flow, flow_gt)
        
        # Combined loss
        total_loss = l1_loss + ms_loss + cs_loss + flow_loss
        
        # Return loss components for logging
        return {
            'total': total_loss,
            'l1': l1_loss,
            'ms': ms_loss,
            'cs': cs_loss,
            'flow': flow_loss
        }