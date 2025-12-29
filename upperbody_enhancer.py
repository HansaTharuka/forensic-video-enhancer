#!/usr/bin/env python3
"""
Upper Body Mask Detection Enhancer
Select upper body area, convert to grayscale, enhance clarity, white background
"""

import cv2
import numpy as np
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

def select_upper_body_area(video_path):
    """Select upper body area interactively"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    print("Select the UPPER BODY area (head to chest/waist)")
    print("Click and drag to select. Press ENTER when done.")
    roi = cv2.selectROI("Select Upper Body Area", frame, False)
    cv2.destroyAllWindows()
    
    return roi if roi[2] > 0 and roi[3] > 0 else None

def enhance_grayscale_clarity(gray_frame):
    """Enhance grayscale image clarity"""
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray_frame, None, 10, 7, 21)
    
    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrasted = clahe.apply(denoised)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(contrasted, -1, kernel)
    
    # Additional edge enhancement
    edges = cv2.Canny(sharpened, 50, 150)
    enhanced = cv2.addWeighted(sharpened, 0.9, edges, 0.1, 0)
    
    return enhanced

def process_video(input_path, roi):
    """Process video with upper body enhancement"""
    x, y, w, h = roi
    output_path = f"upperbody_enhanced_{input_path}"
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Create white background
        white_bg = np.ones_like(frame) * 255
        
        # Extract upper body area
        upper_body = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray_upper_body = cv2.cvtColor(upper_body, cv2.COLOR_BGR2GRAY)
        
        # Enhance clarity
        enhanced_gray = enhance_grayscale_clarity(gray_upper_body)
        
        # Convert back to 3-channel for video
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # Place enhanced area on white background
        white_bg[y:y+h, x:x+w] = enhanced_bgr
        
        # Draw red border around enhanced area
        cv2.rectangle(white_bg, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(white_bg, 'ENHANCED UPPER BODY', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        out.write(white_bg)
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"Enhanced video saved as: {output_path}")
    return output_path

def main():
    print("👤 Upper Body Mask Detection Enhancer")
    print("=" * 40)
    print("Features:")
    print("- Select upper body area")
    print("- Convert to grayscale")
    print("- Enhance clarity")
    print("- White background")
    print("=" * 40)
    
    # Select video
    video_path = select_video()
    if not video_path:
        print("No video selected")
        return
    
    # Select upper body area
    roi = select_upper_body_area(video_path)
    if not roi:
        print("No area selected")
        return
    
    print(f"Upper body area: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    
    # Process video
    output = process_video(video_path, roi)
    print(f"✅ Done! Enhanced video: {output}")

if __name__ == "__main__":
    main()