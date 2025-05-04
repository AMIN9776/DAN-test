import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
import time
from datetime import datetime

from models.dan_model import DAN
from utils.losses import DANLoss
from utils.metrics import calculate_psnr, calculate_ssim

class FisheyeDataset(Dataset):
    def __init__(self, distorted_dir, original_dir, flow_dir=None, transform=None):
        """
        Dataset for fisheye image rectification.
        
        Args:
            distorted_dir: Directory with distorted fisheye images
            original_dir: Directory with original undistorted images
            flow_dir: Directory with ground truth flow maps (optional)
            transform: Image transformations
        """
        self.distorted_dir = distorted_dir
        self.original_dir = original_dir
        self.flow_dir = flow_dir
        self.transform = transform
        
        # Get list of distorted image files
        self.image_files = [f for f in os.listdir(distorted_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get distorted image
        dist_img_name = self.image_files[idx]
        dist_img_path = os.path.join(self.distorted_dir, dist_img_name)
        
        # Parse original image name from distorted image name
        parts = dist_img_name.split('_distorted_')
        orig_img_name = parts[0] + '.png'  # Assuming original images are PNG
        orig_img_path = os.path.join(self.original_dir, orig_img_name)
        
        # Load images
        distorted_img = cv2.imread(dist_img_path)
        distorted_img = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB)
        
        original_img = cv2.imread(orig_img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            distorted_img = self.transform(distorted_img)
            original_img = self.transform(original_img)
        
        # Load flow map if available
        flow = None
        if self.flow_dir:
            flow_name = parts[0] + '_flow_' + parts[1].split('.')[0] + '.npy'
            flow_path = os.path.join(self.flow_dir, flow_name)
            if os.path.exists(flow_path):
                flow = np.load(flow_path)
                flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        
        return {
            'distorted': distorted_img,
            'original': original_img,
            'flow': flow,
            'name': dist_img_name
        }

def train(args):
    # Create directories for saving models and results
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size))
    ])
    
    # Create datasets and data loaders
    train_dataset = FisheyeDataset(
        os.path.join(args.data_dir, 'train', 'distorted'),
        os.path.join(args.data_dir, 'train', 'original'),
        os.path.join(args.data_dir, 'train', 'flow') if args.use_flow else None,
        transform=transform
    )
    
    val_dataset = FisheyeDataset(
        os.path.join(args.data_dir, 'val', 'distorted'),
        os.path.join(args.data_dir, 'val', 'original'),
        os.path.join(args.data_dir, 'val', 'flow') if args.use_flow else None,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = DAN(in_channels=3)
    model = model.to(device)
    
    # Create loss function and optimizer
    criterion = DANLoss(multi_scale_decay=args.decay, num_scales=args.num_scales)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} (Train)') as t:
            for batch in t:
                distorted = batch['distorted'].to(device)
                original = batch['original'].to(device)
                flow = batch['flow'].to(device) if batch['flow'] is not None else None
                
                # Forward pass
                outputs = model(distorted)
                
                # Compute loss
                loss_dict = criterion(outputs, original, flow)
                loss = loss_dict['total']
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} (Val)') as t:
                for batch in t:
                    distorted = batch['distorted'].to(device)
                    original = batch['original'].to(device)
                    flow = batch['flow'].to(device) if batch['flow'] is not None else None
                    
                    # Forward pass
                    outputs = model(distorted)
                    
                    # Compute loss
                    loss_dict = criterion(outputs, original, flow)
                    loss = loss_dict['total']
                    
                    # Compute metrics
                    psnr_val = calculate_psnr(outputs['rectified'][0].cpu(), original[0].cpu())
                    ssim_val = calculate_ssim(outputs['rectified'][0].cpu(), original[0].cpu())
                    
                    # Update statistics
                    val_loss += loss.item()
                    val_psnr += psnr_val
                    val_ssim += ssim_val
                    
                    t.set_postfix(loss=loss.item(), psnr=psnr_val, ssim=ssim_val)
        
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            }, model_path)
            print(f'  Saved best model to {model_path}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            }, checkpoint_path)
            print(f'  Saved checkpoint to {checkpoint_path}')
    
    # Print training summary
    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.2f} seconds')
    print(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DAN for fisheye image rectification')
    parser.add_argument('--data_dir', required=True, help='Directory with processed data')
    parser.add_argument('--save_dir', default='models/saved_models', help='Directory to save models')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--decay', type=float, default=0.9, help='Weight decay factor for multi-scale loss')
    parser.add_argument('--num_scales', type=int, default=5, help='Number of scales for multi-scale loss')
    parser.add_argument('--save_interval', type=int, default=10, help='Epoch interval for saving checkpoints')
    parser.add_argument('--use_flow', action='store_true', help='Use ground truth flow for training')
    
    args = parser.parse_args()
    train(args)