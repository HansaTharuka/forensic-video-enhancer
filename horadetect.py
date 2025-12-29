"""
Forensic Video Enhancement Tool with Region of Interest (ROI) Selection
Focus enhancement on specific areas like faces, hands, or suspicious activity
"""

# Installation cell - Run this first
"""
!pip install opencv-python-headless
!pip install scikit-image
!pip install numpy
!pip install pillow
!pip install moviepy
!pip install matplotlib
"""

import cv2
import numpy as np
from skimage import exposure, restoration
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import glob
import os

class ForensicROIEnhancer:
    def __init__(self):
        self.roi_coords = None
        
    def select_roi_interactive(self, video_path, frame_number=30):
        """
        Interactive ROI selection from a specific frame
        Returns: (x, y, width, height)
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Could not read frame")
            return None
        
        print("🖱️ Click and drag to select the region to enhance")
        print("Press ENTER when done, or 'c' to cancel")
        
        # Let user select ROI
        roi = cv2.selectROI("Select Region of Interest", frame, fromCenter=False)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:
            self.roi_coords = roi
            print(f"✓ ROI Selected: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            return roi
        else:
            print("❌ No ROI selected")
            return None
    
    def select_roi_manual(self, x, y, width, height):
        """
        Manually specify ROI coordinates
        """
        self.roi_coords = (x, y, width, height)
        print(f"✓ ROI Set: x={x}, y={y}, w={width}, h={height}")
        return self.roi_coords
    
    def preview_roi(self, video_path, frame_number=30):
        """
        Show the selected ROI on a frame
        """
        if self.roi_coords is None:
            print("❌ No ROI selected yet")
            return
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return
        
        x, y, w, h = self.roi_coords
        
        # Draw ROI on frame
        preview = frame.copy()
        cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(preview, 'ENHANCEMENT AREA', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        ax.set_title('Selected Region for Enhancement', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def extreme_denoise(self, frame, strength=15):
        """Ultra-strong denoising for very noisy footage"""
        return cv2.fastNlMeansDenoisingColored(
            frame, None, h=strength, hColor=strength,
            templateWindowSize=7, searchWindowSize=21
        )
    
    def enhance_details_extreme(self, frame):
        """Extreme detail enhancement"""
        # Multi-scale sharpening
        kernel1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp1 = cv2.filter2D(frame, -1, kernel1)
        
        # Unsharp mask
        gaussian = cv2.GaussianBlur(frame, (0, 0), 3.0)
        sharp2 = cv2.addWeighted(frame, 2.0, gaussian, -1.0, 0)
        
        # Combine
        result = cv2.addWeighted(sharp1, 0.5, sharp2, 0.5, 0)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def enhance_contrast_extreme(self, frame):
        """Extreme contrast enhancement"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aggressive CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        
        # Additional histogram equalization
        l = cv2.equalizeHist(l)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def enhance_dark_areas(self, frame):
        """Specifically enhance dark/shadow areas"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Gamma correction for dark areas
        v_float = v.astype(float) / 255.0
        v_corrected = np.power(v_float, 0.5)  # Gamma < 1 brightens
        v = (v_corrected * 255).astype(np.uint8)
        
        enhanced = cv2.merge([h, s, v])
        return cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
    
    def super_enhance_roi(self, roi_frame, scale=3):
        """
        Super enhancement specifically for ROI
        Includes: upscaling, denoising, sharpening, contrast
        """
        h, w = roi_frame.shape[:2]
        
        # 1. Upscale
        upscaled = cv2.resize(roi_frame, (w * scale, h * scale), 
                             interpolation=cv2.INTER_CUBIC)
        
        # 2. Denoise
        denoised = self.extreme_denoise(upscaled, strength=12)
        
        # 3. Enhance dark areas
        brightened = self.enhance_dark_areas(denoised)
        
        # 4. Extreme contrast
        contrasted = self.enhance_contrast_extreme(brightened)
        
        # 5. Extreme sharpening
        sharpened = self.enhance_details_extreme(contrasted)
        
        # 6. Edge enhancement
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(sharpened, 0.9, edges_colored, 0.1, 0)
        
        return result
    
    def process_video_with_roi(self, input_path, output_path,
                               roi_coords=None, 
                               scale_roi=3,
                               white_background=True,
                               show_roi_only=False):
        """
        Process video with focus on ROI
        
        Parameters:
        -----------
        roi_coords: tuple (x, y, width, height) - Region to enhance
        scale_roi: int - Upscaling factor for ROI (2-4 recommended)
        white_background: bool - Show rest of frame as white
        show_roi_only: bool - Only show the ROI, cropped
        """
        if roi_coords is None:
            if self.roi_coords is None:
                print("❌ No ROI specified. Use select_roi_interactive() or select_roi_manual()")
                return None
            roi_coords = self.roi_coords
        
        x, y, w, h = roi_coords
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate output dimensions
        if show_roi_only:
            out_width = w * scale_roi
            out_height = h * scale_roi
        else:
            out_width = orig_width
            out_height = orig_height
        
        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        frame_count = 0
        
        print(f"{'='*70}")
        print(f"🎬 Processing Video with ROI Focus")
        print(f"{'='*70}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Total Frames: {total_frames}")
        print(f"ROI: x={x}, y={y}, w={w}, h={h}")
        print(f"Scale Factor: {scale_roi}x")
        print(f"Mode: {'ROI ONLY' if show_roi_only else 'FULL FRAME with ROI highlight'}")
        print(f"{'='*70}\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w].copy()
            
            # Super enhance the ROI
            enhanced_roi = self.super_enhance_roi(roi, scale=scale_roi)
            
            if show_roi_only:
                # Output only the enhanced ROI
                output_frame = enhanced_roi
            else:
                # Create output frame
                if white_background:
                    output_frame = np.ones_like(frame) * 255
                else:
                    output_frame = frame.copy()
                
                # Resize enhanced ROI back to original size for overlay
                enhanced_roi_resized = cv2.resize(enhanced_roi, (w, h))
                
                # Place enhanced ROI back
                output_frame[y:y+h, x:x+w] = enhanced_roi_resized
                
                # Draw red border around ROI
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                
                # Add label
                cv2.putText(output_frame, 'ENHANCED AREA', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            out.write(output_frame)
            
            if frame_count % 30 == 0:
                print(f"⏳ Processed {frame_count}/{total_frames} frames "
                      f"({100*frame_count/total_frames:.1f}%)")
        
        cap.release()
        out.release()
        
        print(f"\n{'='*70}")
        print("✅ Processing Complete!")
        print(f"📁 Output saved to: {output_path}")
        print(f"📊 Total frames processed: {frame_count}")
        print(f"{'='*70}")
        
        return output_path
    
    def show_comparison(self, input_path, output_path, frame_num=30):
        """Display before/after comparison"""
        cap_in = cv2.VideoCapture(input_path)
        cap_out = cv2.VideoCapture(output_path)
        
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        cap_out.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        ret_in, frame_in = cap_in.read()
        ret_out, frame_out = cap_out.read()
        
        if ret_in and ret_out:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            axes[0].imshow(cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Frame', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Enhanced (ROI Focused)', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Also show zoomed ROI if available
            if self.roi_coords:
                x, y, w, h = self.roi_coords
                roi_original = frame_in[y:y+h, x:x+w]
                
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
                axes[0].imshow(cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB))
                axes[0].set_title('Original ROI (Zoomed)', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                if frame_out.shape == frame_in.shape:
                    roi_enhanced = frame_out[y:y+h, x:x+w]
                else:
                    roi_enhanced = frame_out
                
                axes[1].imshow(cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2RGB))
                axes[1].set_title('Enhanced ROI (Zoomed)', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.show()
        
        cap_in.release()
        cap_out.release()

# ============================================================================
# MAIN USAGE FUNCTIONS
# ============================================================================

def get_video_files():
    """Get list of video files in current directory"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(ext))
    return video_files

def select_video_file():
    """Interactive video file selection"""
    video_files = get_video_files()
    
    if not video_files:
        print("❌ No video files found in current directory")
        print("Please place your video file in the same directory as this script")
        return None
    
    print("📁 Available video files:")
    for i, file in enumerate(video_files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = int(input("\nSelect video file (number): ")) - 1
            if 0 <= choice < len(video_files):
                return video_files[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def process_with_interactive_selection(video_path):
    """
    Interactive mode: Click and drag to select area
    """
    enhancer = ForensicROIEnhancer()
    
    print("📹 Step 1: Selecting ROI from video...")
    roi = enhancer.select_roi_interactive(video_path, frame_number=30)
    
    if roi is None:
        print("❌ ROI selection cancelled")
        return
    
    print("\n📋 Step 2: Preview selected region...")
    enhancer.preview_roi(video_path, frame_number=30)
    
    output_path = 'enhanced_roi_' + os.path.basename(video_path)
    
    print("\n⚙️ Step 3: Processing video...")
    print("Choose mode:")
    print("1. Show only enhanced ROI (recommended for person identification)")
    print("2. Show full frame with white background")
    
    mode = input("Enter 1 or 2: ").strip()
    
    enhancer.process_video_with_roi(
        input_path=video_path,
        output_path=output_path,
        scale_roi=3,  # 3x upscaling
        white_background=True,
        show_roi_only=(mode == '1')
    )
    
    print("\n📊 Step 4: Showing comparison...")
    enhancer.show_comparison(video_path, output_path, frame_num=30)
    
    return output_path

def process_with_manual_coordinates(video_path, x, y, width, height, roi_only=True):
    """
    Manual mode: Specify exact coordinates
    
    Example for your case (person at door):
    x=400, y=300, width=200, height=400
    """
    enhancer = ForensicROIEnhancer()
    
    print(f"📍 Setting ROI: x={x}, y={y}, w={width}, h={height}")
    enhancer.select_roi_manual(x, y, width, height)
    
    print("\n📋 Preview selected region...")
    enhancer.preview_roi(video_path, frame_number=30)
    
    output_path = 'enhanced_roi_' + os.path.basename(video_path)
    
    print("\n⚙️ Processing video with extreme enhancement...")
    enhancer.process_video_with_roi(
        input_path=video_path,
        output_path=output_path,
        scale_roi=4,  # 4x upscaling for maximum detail
        white_background=True,
        show_roi_only=roi_only
    )
    
    print("\n📊 Showing comparison...")
    enhancer.show_comparison(video_path, output_path, frame_num=30)
    
    return output_path

# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    print("""
    🎥 FORENSIC ROI VIDEO ENHANCEMENT TOOL
    ═══════════════════════════════════════
    
    This tool allows you to:
    1. Select a specific area (ROI) to enhance
    2. Apply extreme enhancement to that area only
    3. Show enhanced area on white background
    4. Focus on faces, hands, or suspicious activity
    
    ═══════════════════════════════════════
    """)
    
    # Select video file
    video_path = select_video_file()
    if video_path is None:
        sys.exit(1)
    
    print(f"\n✓ Video selected: {video_path}")
    print("\nChoose selection method:")
    print("1. Interactive (click and drag)")
    print("2. Manual (enter coordinates)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        process_with_interactive_selection(video_path)
    else:
        print("\nEnter ROI coordinates:")
        x = int(input("X (left edge): "))
        y = int(input("Y (top edge): "))
        width = int(input("Width: "))
        height = int(input("Height: "))
        
        print("\nShow only enhanced ROI? (recommended for face/body analysis)")
        roi_only = input("Y/N: ").strip().upper() == 'Y'
        
        process_with_manual_coordinates(video_path, x, y, width, height, roi_only)