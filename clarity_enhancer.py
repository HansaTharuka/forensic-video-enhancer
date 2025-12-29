#!/usr/bin/env python3
"""
Video Clarity Enhancer - Focus on selected area with grain reduction
"""

import cv2
import numpy as np
import glob
import sys

def get_video_files():
    """Find video files in current directory"""
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

def select_roi(video_path):
    """Interactive ROI selection"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    print("Click and drag to select area. Press ENTER when done.")
    roi = cv2.selectROI("Select Area to Enhance", frame, False)
    cv2.destroyAllWindows()
    
    return roi if roi[2] > 0 and roi[3] > 0 else None

def enhance_clarity(frame):
    """Enhance clarity and reduce grain"""
    # Denoise first
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    # Sharpen for clarity
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Enhance contrast
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return result

def process_video(input_path, roi):
    """Process video with ROI enhancement"""
    x, y, w, h = roi
    output_path = f"enhanced_{input_path}"
    
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
        
        # Extract and enhance ROI
        roi_area = frame[y:y+h, x:x+w]
        enhanced_roi = enhance_clarity(roi_area)
        
        # Replace ROI in original frame
        result_frame = frame.copy()
        result_frame[y:y+h, x:x+w] = enhanced_roi
        
        # Draw border around enhanced area
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        out.write(result_frame)
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"Enhanced video saved as: {output_path}")
    return output_path

def main():
    print("🎥 Video Clarity Enhancer")
    print("=" * 30)
    
    # Select video
    video_path = select_video()
    if not video_path:
        print("No video selected")
        return
    
    # Select ROI
    roi = select_roi(video_path)
    if not roi:
        print("No area selected")
        return
    
    print(f"Selected area: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    
    # Process video
    output = process_video(video_path, roi)
    print(f"✅ Done! Output: {output}")

if __name__ == "__main__":
    main()