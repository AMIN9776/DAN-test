import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.dan_model import DAN
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpm, visualize_flow

def create_results_visualization(distorted, rectified, gt, save_path):
    """Create a visualization of the results"""
    # Convert tensors to numpy if needed
    if torch.is_tensor(distorted):
        distorted = distorted.detach().cpu().numpy().transpose(1, 2, 0)
    if torch.is_tensor(rectified):
        rectified = rectified.detach().cpu().numpy().transpose(1, 2, 0)
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Ensure all values are in [0, 1]
    distorted = np.clip(distorted, 0, 1)
    rectified = np.clip(rectified, 0, 1)
    gt = np.clip(gt, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot images
    axes[0].imshow(distorted)
    axes[0].set_title('Distorted')
    axes[0].axis('off')
    
    axes[1].imshow(rectified)
    axes[1].set_title('Rectified (Ours)')
    axes[1].axis('off')
    
    axes[2].imshow(gt)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def test(args):
    """
    Test the DAN model on a dataset.
    """
    # Create directory for saving results
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size))
    ])
    
    # Create dataset and data loader
    if args.real_fisheye:
        # For real fisheye images without ground truth
        class RealFisheyeDataset(Dataset):
            def __init__(self, img_dir, transform=None):
                self.img_dir = img_dir
                self.transform = transform
                self.image_files = [f for f in os.listdir(img_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
            def __len__(self):
                return len(self.image_files)
            
            def __getitem__(self, idx):
                img_name = self.image_files[idx]
                img_path = os.path.join(self.img_dir, img_name)
                
                # Load image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                if self.transform:
                    img = self.transform(img)
                
                return {
                    'distorted': img,
                    'name': img_name
                }
        
        dataset = RealFisheyeDataset(args.test_dir, transform=transform)
    else:
        # For synthetic dataset with ground truth
        class SyntheticTestDataset(Dataset):
            def __init__(self, distorted_dir, original_dir, transform=None):
                self.distorted_dir = distorted_dir
                self.original_dir = original_dir
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
                
                return {
                    'distorted': distorted_img,
                    'original': original_img,
                    'name': dist_img_name
                }
        
        dataset = SyntheticTestDataset(
            os.path.join(args.test_dir, 'distorted'),
            os.path.join(args.test_dir, 'original'),
            transform=transform
        )
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Load model
    model = DAN(in_channels=3)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Metrics
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpm = 0.0
    
    # Testing loop
    with torch.no_grad():
        with tqdm(data_loader, desc='Testing') as t:
            for i, batch in enumerate(t):
                distorted = batch['distorted'].to(device)
                
                # Forward pass
                outputs = model(distorted)
                rectified = outputs['rectified']
                
                # Save results
                img_name = batch['name'][0]
                save_path = os.path.join(args.results_dir, f'rectified_{img_name}')
                
                # Save rectified image
                rectified_np = rectified[0].cpu().numpy().transpose(1, 2, 0)
                rectified_np = np.clip(rectified_np * 255, 0, 255).astype(np.uint8)
                rectified_rgb = cv2.cvtColor(rectified_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, rectified_rgb)
                
                # Compute and save flow visualization
                flow = outputs['flows']['flow256'][0].cpu().numpy().transpose(1, 2, 0)
                flow_vis = visualize_flow(flow)
                flow_path = os.path.join(args.results_dir, f'flow_{img_name}')
                cv2.imwrite(flow_path, flow_vis)
                
                # Compute metrics for synthetic dataset
                if not args.real_fisheye:
                    original = batch['original'].to(device)
                    
                    # Calculate metrics
                    psnr_val = calculate_psnr(rectified[0].cpu(), original[0].cpu())
                    ssim_val = calculate_ssim(rectified[0].cpu(), original[0].cpu())
                    
                    # Update statistics
                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    
                    # Create visualization
                    vis_path = os.path.join(args.results_dir, f'compare_{img_name}')
                    create_results_visualization(
                        distorted[0].cpu(), 
                        rectified[0].cpu(), 
                        original[0].cpu(), 
                        vis_path
                    )
                    
                    t.set_postfix(psnr=psnr_val, ssim=ssim_val)
                
                # Calculate LPM for both real and synthetic datasets
                lpm_val = calculate_lpm(rectified[0].cpu())
                total_lpm += lpm_val
    
    # Print average metrics
    if not args.real_fisheye:
        avg_psnr = total_psnr / len(data_loader)
        avg_ssim = total_ssim / len(data_loader)
        print(f'Average PSNR: {avg_psnr:.2f}')
        print(f'Average SSIM: {avg_ssim:.4f}')
    
    avg_lpm = total_lpm / len(data_loader)
    print(f'Average LPM: {avg_lpm:.4f} (lower is better)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DAN for fisheye image rectification')
    parser.add_argument('--test_dir', required=True, help='Directory with test data')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--results_dir', default='results', help='Directory to save results')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--real_fisheye', action='store_true', help='Test on real fisheye images without ground truth')
    
    args = parser.parse_args()
    test(args)