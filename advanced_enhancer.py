#!/usr/bin/env python3
"""
Advanced Feature Enhancement with Shadow Reduction
Uses advanced image processing to highlight all features and lines
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, filters
import glob

def get_video_files():
    """Find video files"""
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    files = []
    for ext in extensions:
        files.extend(glob.glob(ext))
    return files

def select_video():
    """Select video file"""
    videos = get_video_files()
    if not videos:
        print("No video files found")
        return None
    
    print("Available videos:")
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video}")
    
    try:
        choice = int(input("Select video: ")) - 1
        return videos[choice] if 0 <= choice < len(videos) else None
    except:
        return None

def select_area(video_path):
    """Select area for enhancement"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    print("Select area for advanced feature enhancement")
    print("Click and drag to select. Press ENTER when done.")
    roi = cv2.selectROI("Select Area for Feature Enhancement", frame, False)
    cv2.destroyAllWindows()
    
    return roi if roi[2] > 0 and roi[3] > 0 else None

def advanced_feature_enhancement(frame):
    """Advanced feature enhancement with shadow reduction"""
    
    # 1. Shadow reduction using gamma correction
    gamma_corrected = np.power(frame / 255.0, 0.5) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # 2. Multi-scale retinex for illumination normalization
    def single_scale_retinex(img, sigma):
        retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
        return retinex
    
    # Convert to float for processing
    img_float = gamma_corrected.astype(np.float64) + 1.0
    
    # Apply multi-scale retinex
    retinex1 = single_scale_retinex(img_float, 15)
    retinex2 = single_scale_retinex(img_float, 80)
    retinex3 = single_scale_retinex(img_float, 250)
    retinex = (retinex1 + retinex2 + retinex3) / 3.0
    
    # Normalize retinex output
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
    retinex = retinex.astype(np.uint8)
    
    # 3. Advanced edge detection and enhancement
    gray = cv2.cvtColor(retinex, cv2.COLOR_BGR2GRAY)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    # Laplacian for fine details
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Combine edge information
    edges_combined = cv2.normalize(sobel + np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges_colored = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)
    
    # 4. Adaptive histogram equalization
    lab = cv2.cvtColor(retinex, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 5. Unsharp masking for detail enhancement
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    # 6. Combine with edge information
    result = cv2.addWeighted(unsharp, 0.85, edges_colored, 0.15, 0)
    
    # 7. Final contrast and brightness adjustment
    alpha = 1.2  # Contrast
    beta = 10    # Brightness
    final = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    # 8. Noise reduction while preserving edges
    denoised = cv2.bilateralFilter(final, 9, 75, 75)
    
    return denoised

def process_video(input_path, roi):
    """Process video with advanced enhancement"""
    x, y, w, h = roi
    output_path = f"advanced_enhanced_{input_path}"
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames with advanced techniques...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Create white background
        white_bg = np.ones_like(frame) * 255
        
        # Extract selected area
        selected_area = frame[y:y+h, x:x+w]
        
        # Apply advanced enhancement
        enhanced_area = advanced_feature_enhancement(selected_area)
        
        # Place enhanced area on white background
        white_bg[y:y+h, x:x+w] = enhanced_area
        
        # Draw blue border around enhanced area
        cv2.rectangle(white_bg, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(white_bg, 'ADVANCED ENHANCED', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        out.write(white_bg)
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"Advanced enhanced video saved as: {output_path}")
    return output_path

def main():
    print("🔬 Advanced Feature Enhancement")
    print("=" * 40)
    print("Techniques used:")
    print("- Multi-scale Retinex (shadow reduction)")
    print("- Advanced edge detection (Sobel + Laplacian)")
    print("- Adaptive histogram equalization")
    print("- Unsharp masking")
    print("- Bilateral filtering")
    print("=" * 40)
    
    # Select video
    video_path = select_video()
    if not video_path:
        print("No video selected")
        return
    
    # Select area
    roi = select_area(video_path)
    if not roi:
        print("No area selected")
        return
    
    print(f"Selected area: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    
    # Process video
    output = process_video(video_path, roi)
    print(f"✅ Done! Advanced enhanced video: {output}")

if __name__ == "__main__":
    main()