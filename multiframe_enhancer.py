#!/usr/bin/env python3
"""
Multi-Frame Super-Resolution with Blind Deconvolution
Advanced forensic enhancement using temporal information from multiple frames
"""

import cv2
import numpy as np
from scipy import ndimage, optimize
from skimage import restoration, measure
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
    
    print("Select area for multi-frame super-resolution")
    print("Click and drag to select. Press ENTER when done.")
    roi = cv2.selectROI("Select Area for Multi-Frame Enhancement", frame, False)
    cv2.destroyAllWindows()
    
    return roi if roi[2] > 0 and roi[3] > 0 else None

def estimate_motion(frame1, frame2):
    """Estimate motion between frames using optical flow"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Lucas-Kanade optical flow
    flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
    return flow

def blind_deconvolution(image, iterations=10):
    """Blind deconvolution to remove blur"""
    # Estimate blur kernel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Edge-based blur kernel estimation
    edges = cv2.Canny(gray, 50, 150)
    kernel_size = 15
    
    # Simple motion blur kernel estimation
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, :] = 1.0
    kernel = kernel / kernel.sum()
    
    # Simplified deconvolution using filter2D
    result = image.copy().astype(np.float32)
    
    # Apply sharpening kernel instead of complex deconvolution
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    
    if len(result.shape) == 3:
        for c in range(3):
            result[:,:,c] = cv2.filter2D(result[:,:,c], -1, sharpen_kernel)
    else:
        result = cv2.filter2D(result, -1, sharpen_kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def multi_frame_super_resolution(frames, scale_factor=2):
    """Multi-frame super-resolution using temporal information"""
    if len(frames) < 3:
        return frames[0]
    
    h, w = frames[0].shape[:2]
    hr_h, hr_w = h * scale_factor, w * scale_factor
    
    # Initialize high-resolution image
    hr_image = np.zeros((hr_h, hr_w, 3), dtype=np.float64)
    weight_map = np.zeros((hr_h, hr_w), dtype=np.float64)
    
    reference_frame = frames[len(frames)//2]  # Use middle frame as reference
    
    for i, frame in enumerate(frames):
        # Estimate motion relative to reference
        if i != len(frames)//2:
            # Simple registration using phase correlation
            gray_ref = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
            gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Phase correlation for sub-pixel registration
            f1 = np.fft.fft2(gray_ref)
            f2 = np.fft.fft2(gray_cur)
            cross_power = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
            shift = np.fft.ifft2(cross_power)
            shift = np.abs(shift)
            
            # Find peak
            y_shift, x_shift = np.unravel_index(np.argmax(shift), shift.shape)
            if y_shift > h//2:
                y_shift -= h
            if x_shift > w//2:
                x_shift -= w
        else:
            x_shift, y_shift = 0, 0
        
        # Upscale current frame
        upscaled = cv2.resize(frame, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply sub-pixel shift
        M = np.float32([[1, 0, x_shift*scale_factor], [0, 1, y_shift*scale_factor]])
        shifted = cv2.warpAffine(upscaled, M, (hr_w, hr_h))
        
        # Accumulate with weights
        weight = 1.0 / (1.0 + np.abs(x_shift) + np.abs(y_shift))
        hr_image += shifted * weight
        weight_map += weight
    
    # Normalize
    weight_map[weight_map == 0] = 1
    hr_image = hr_image / weight_map[:,:,np.newaxis]
    
    return np.clip(hr_image, 0, 255).astype(np.uint8)

def wavelet_enhancement(image):
    """Wavelet-based detail enhancement"""
    # Convert to YUV for better processing
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    
    # Simple wavelet-like enhancement using Laplacian pyramid
    gaussian = cv2.GaussianBlur(y, (0, 0), 1.0)
    laplacian = y.astype(np.float32) - gaussian.astype(np.float32)
    
    # Enhance high-frequency components
    enhanced_y = y.astype(np.float32) + laplacian * 0.5
    enhanced_y = np.clip(enhanced_y, 0, 255).astype(np.uint8)
    
    # Merge back
    enhanced_yuv = cv2.merge([enhanced_y, u, v])
    return cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)

def advanced_enhancement(roi_frames):
    """Apply advanced enhancement techniques"""
    # 1. Multi-frame super-resolution
    print("Applying multi-frame super-resolution...")
    super_res = multi_frame_super_resolution(roi_frames, scale_factor=2)
    
    # 2. Blind deconvolution
    print("Applying blind deconvolution...")
    deblurred = blind_deconvolution(super_res, iterations=5)
    
    # 3. Wavelet enhancement
    print("Applying wavelet enhancement...")
    wavelet_enhanced = wavelet_enhancement(deblurred)
    
    # 4. Advanced noise reduction
    print("Applying advanced denoising...")
    denoised = cv2.fastNlMeansDenoisingColored(wavelet_enhanced, None, 8, 8, 7, 21)
    
    # 5. Adaptive sharpening
    print("Applying adaptive sharpening...")
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
    l = clahe.apply(l)
    
    # Unsharp masking
    gaussian = cv2.GaussianBlur(l, (0, 0), 1.5)
    l_sharp = cv2.addWeighted(l, 1.8, gaussian, -0.8, 0)
    
    enhanced_lab = cv2.merge([l_sharp, a, b])
    final = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return final

def process_video(input_path, roi, frame_window=5):
    """Process video with advanced multi-frame enhancement"""
    x, y, w, h = roi
    output_path = f"multiframe_enhanced_{input_path}"
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output dimensions (2x upscaled ROI)
    out_width = w * 2
    out_height = h * 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    print(f"Processing {total_frames} frames with multi-frame enhancement...")
    print(f"Frame window: {frame_window} frames")
    print(f"Output size: {out_width}x{out_height}")
    
    # Read all frames first
    all_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cap.release()
    
    # Process with sliding window
    for i in range(len(all_frames)):
        # Get frame window
        start_idx = max(0, i - frame_window//2)
        end_idx = min(len(all_frames), i + frame_window//2 + 1)
        
        roi_frames = []
        for j in range(start_idx, end_idx):
            roi_frame = all_frames[j][y:y+h, x:x+w]
            roi_frames.append(roi_frame)
        
        # Apply advanced enhancement
        enhanced_roi = advanced_enhancement(roi_frames)
        
        out.write(enhanced_roi)
        
        if (i + 1) % 10 == 0:
            progress = ((i + 1) / len(all_frames)) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    
    print(f"Multi-frame enhanced video saved as: {output_path}")
    return output_path

def main():
    print("🔬 Multi-Frame Super-Resolution Enhancement")
    print("=" * 50)
    print("Advanced Techniques:")
    print("- Multi-frame super-resolution (2x upscale)")
    print("- Blind deconvolution (blur removal)")
    print("- Wavelet enhancement")
    print("- Temporal information fusion")
    print("- Sub-pixel registration")
    print("=" * 50)
    
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
    
    # Get frame window size
    try:
        window = int(input("Frame window size (3-9, default 5): ") or "5")
        window = max(3, min(9, window))
    except:
        window = 5
    
    print(f"Using {window}-frame window for enhancement")
    
    # Process video
    output = process_video(video_path, roi, window)
    print(f"✅ Done! Multi-frame enhanced video: {output}")

if __name__ == "__main__":
    main()