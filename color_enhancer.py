#!/usr/bin/env python3
"""
Color Enhancement with Shadow Reduction
Select area, enhance colors, reduce shadows, keep colored output
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

def select_area(video_path):
    """Select area to enhance"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    print("Select area to enhance colors and reduce shadows")
    print("Click and drag to select. Press ENTER when done.")
    roi = cv2.selectROI("Select Area for Color Enhancement", frame, False)
    cv2.destroyAllWindows()
    
    return roi if roi[2] > 0 and roi[3] > 0 else None

def enhance_colors_reduce_shadows(frame):
    """Enhance colors and reduce shadows"""
    # Convert to HSV for better color control
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Brighten shadows (gamma correction)
    v_float = v.astype(float) / 255.0
    v_corrected = np.power(v_float, 0.6)  # Brighten dark areas
    v = (v_corrected * 255).astype(np.uint8)
    
    # Enhance saturation for vivid colors
    s = cv2.multiply(s, 1.3)
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # Merge back
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    # Apply CLAHE to reduce harsh shadows
    lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    final_lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
    
    # Slight sharpening to maintain detail
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(result, -1, kernel)
    
    return sharpened

def process_video(input_path, roi):
    """Process video with color enhancement"""
    x, y, w, h = roi
    output_path = f"color_enhanced_{input_path}"
    
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
        
        # Extract selected area
        selected_area = frame[y:y+h, x:x+w]
        
        # Enhance colors and reduce shadows
        enhanced_area = enhance_colors_reduce_shadows(selected_area)
        
        # Place enhanced area on white background
        white_bg[y:y+h, x:x+w] = enhanced_area
        
        # Draw green border around enhanced area
        cv2.rectangle(white_bg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(white_bg, 'COLOR ENHANCED', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(white_bg)
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"Enhanced video saved as: {output_path}")
    return output_path

def main():
    print("🌈 Color Enhancement with Shadow Reduction")
    print("=" * 45)
    print("Features:")
    print("- Enhance colors and saturation")
    print("- Reduce shadows and dark areas")
    print("- Maintain colored output")
    print("- White background")
    print("=" * 45)
    
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
    print(f"✅ Done! Enhanced video: {output}")

if __name__ == "__main__":
    main()