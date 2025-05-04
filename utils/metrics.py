import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images.
    
    Args:
        img1, img2: Images to compare (numpy arrays or tensors)
        
    Returns:
        PSNR value
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    return psnr_func(img1, img2)

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.
    
    Args:
        img1, img2: Images to compare (numpy arrays or tensors)
        
    Returns:
        SSIM value
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    return ssim_func(img1, img2, multichannel=True)

def calculate_lpm(img):
    """
    Calculate Line Preservation Metric (LPM) as described in the paper.
    
    Args:
        img: Image to evaluate
        
    Returns:
        LPM value (lower is better)
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Convert to grayscale
    if img.shape[-1] == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255).astype(np.uint8)
    
    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Extract lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is None:
        return 0.0  # No lines detected
    
    total_deviation = 0.0
    num_lines = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip near-horizontal or near-vertical lines (can cause division by zero)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            continue
        
        # Calculate target slope
        target_slope = (y2 - y1) / (x2 - x1)
        
        # Divide line into n segments
        n = 10  # Number of segments
        points = []
        for i in range(n + 1):
            t = i / n
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            points.append((x, y))
        
        # Calculate slope deviation for each segment
        segment_deviations = 0.0
        num_segments = 0
        
        for i in range(1, n + 1):
            x_prev, y_prev = points[i-1]
            x_curr, y_curr = points[i]
            
            # Skip segments that are too short
            if abs(x_curr - x_prev) < 3:
                continue
            
            segment_slope = (y_curr - y_prev) / (x_curr - x_prev)
            segment_deviations += abs(segment_slope - target_slope)
            num_segments += 1
        
        if num_segments > 0:
            avg_deviation = segment_deviations / num_segments
            total_deviation += avg_deviation
            num_lines += 1
    
    if num_lines == 0:
        return 0.0
    
    # Average deviation across all lines (lower is better)
    return total_deviation / num_lines

def visualize_flow(flow):
    """Create a visualization of the flow map"""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return rgb