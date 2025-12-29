#!/usr/bin/env python3
"""
Facial Feature Enhancement for Masked Person
Select face area, enhance features, zoom without quality loss, clean colored output
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

def select_face_area(video_path):
    """Select face area interactively"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    print("Select the FACE area (including forehead and visible features)")
    print("Click and drag to select. Press ENTER when done.")
    roi = cv2.selectROI("Select Face Area", frame, False)
    cv2.destroyAllWindows()
    
    return roi if roi[2] > 0 and roi[3] > 0 else None

def enhance_facial_features(face_frame, zoom_factor=3):
    """Enhance facial features with high-quality zoom"""
    h, w = face_frame.shape[:2]
    
    # High-quality upscaling using INTER_CUBIC
    zoomed = cv2.resize(face_frame, (w * zoom_factor, h * zoom_factor), 
                       interpolation=cv2.INTER_CUBIC)
    
    # Advanced denoising while preserving details
    denoised = cv2.fastNlMeansDenoisingColored(zoomed, None, 8, 8, 7, 21)
    
    # Enhance eye area and visible features
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Subtle sharpening for facial features
    kernel = np.array([[0,-0.5,0], [-0.5,3,-0.5], [0,-0.5,0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Edge enhancement for visible features (eyes, eyebrows)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 80)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    final = cv2.addWeighted(sharpened, 0.95, edges_colored, 0.05, 0)
    
    # Color enhancement for natural look
    hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = cv2.multiply(s_ch, 1.1)  # Slight saturation boost
    enhanced_hsv = cv2.merge([h_ch, s_ch, v_ch])
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    return result

def process_video(input_path, face_roi, zoom_factor=3):
    """Process video with facial enhancement"""
    x, y, w, h = face_roi
    output_path = f"face_enhanced_{input_path}"
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output dimensions based on zoomed face
    out_width = w * zoom_factor
    out_height = h * zoom_factor
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    print(f"Processing {total_frames} frames...")
    print(f"Output size: {out_width}x{out_height} (zoomed {zoom_factor}x)")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Extract face area
        face_area = frame[y:y+h, x:x+w]
        
        # Enhance and zoom face
        enhanced_face = enhance_facial_features(face_area, zoom_factor)
        
        # Output only the enhanced face (zoomed)
        out.write(enhanced_face)
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"Enhanced face video saved as: {output_path}")
    return output_path

def main():
    print("👤 Facial Feature Enhancement for Masked Person")
    print("=" * 50)
    print("Features:")
    print("- High-quality face zoom (3x default)")
    print("- Enhance visible facial features")
    print("- Clean colored output")
    print("- Preserve eye and forehead details")
    print("=" * 50)
    
    # Select video
    video_path = select_video()
    if not video_path:
        print("No video selected")
        return
    
    # Select face area
    face_roi = select_face_area(video_path)
    if not face_roi:
        print("No face area selected")
        return
    
    print(f"Face area: x={face_roi[0]}, y={face_roi[1]}, w={face_roi[2]}, h={face_roi[3]}")
    
    # Get zoom factor
    try:
        zoom = int(input("Enter zoom factor (2-5, default 3): ") or "3")
        zoom = max(2, min(5, zoom))  # Limit between 2-5
    except:
        zoom = 3
    
    print(f"Using {zoom}x zoom")
    
    # Process video
    output = process_video(video_path, face_roi, zoom)
    print(f"✅ Done! Enhanced face video: {output}")

if __name__ == "__main__":
    main()