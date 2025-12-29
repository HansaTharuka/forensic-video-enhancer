# Video Enhancement Tools

A collection of specialized video enhancement scripts for forensic analysis, clarity improvement, and feature detection.

## Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd forensic-video-enhancer
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Video Files
Place your video files (`.mp4`, `.avi`, `.mov`, `.mkv`) in the project directory.

## How to Run

### Basic Usage
```bash
# Activate virtual environment
source .venv/bin/activate

# Run any script
python [script_name].py
```

### Example Workflow
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run facial enhancement
python face_enhancer.py

# 3. Follow interactive prompts:
#    - Select video from list
#    - Click and drag to select face area
#    - Choose zoom factor (2-5x)
#    - Wait for processing
```

## Scripts Overview

### 1. `horadetect.py` - Forensic ROI Video Enhancement
**Purpose**: Comprehensive forensic video enhancement with Region of Interest (ROI) selection.

**Features**:
- Interactive ROI selection (click and drag)
- Manual coordinate input
- Extreme enhancement with upscaling (2x-4x)
- White background option
- Before/after comparison

**Usage**:
```bash
python horadetect.py
```

**Output**: `enhanced_roi_[filename].mp4`

---

### 2. `clarity_enhancer.py` - Basic Clarity Enhancement
**Purpose**: Simple clarity enhancement with grain reduction for selected areas.

**Features**:
- Interactive area selection
- Denoising and sharpening
- Contrast enhancement
- Green border marking

**Usage**:
```bash
python clarity_enhancer.py
```

**Output**: `enhanced_[filename].mp4`

---

### 3. `upperbody_enhancer.py` - Upper Body Analysis
**Purpose**: Specialized for analyzing masked individuals' upper body areas.

**Features**:
- Upper body area selection
- Grayscale conversion
- Clarity enhancement with grain reduction
- White background
- Red border marking

**Usage**:
```bash
python upperbody_enhancer.py
```

**Output**: `upperbody_enhanced_[filename].mp4`

---

### 4. `color_enhancer.py` - Color Enhancement with Shadow Reduction
**Purpose**: Enhance colors and reduce shadows while maintaining colored output.

**Features**:
- HSV color space manipulation
- Shadow reduction via gamma correction
- Saturation boost (1.3x)
- CLAHE for shadow reduction
- Colored output with white background

**Usage**:
```bash
python color_enhancer.py
```

**Output**: `color_enhanced_[filename].mp4`

---

### 5. `face_enhancer.py` - Facial Feature Enhancement
**Purpose**: High-quality facial feature enhancement for masked individuals.

**Features**:
- Face area selection
- High-quality zoom (2x-5x) using INTER_CUBIC
- Eye and forehead detail enhancement
- Clean colored output
- Face-only video output

**Usage**:
```bash
python face_enhancer.py
```

**Output**: `face_enhanced_[filename].mp4`

---

### 6. `advanced_enhancer.py` - Advanced Image Processing
**Purpose**: Professional-grade enhancement using advanced image processing techniques.

**Features**:
- Multi-scale Retinex for shadow reduction
- Advanced edge detection (Sobel + Laplacian)
- Adaptive histogram equalization
- Unsharp masking
- Bilateral filtering
- Feature and line highlighting

**Usage**:
```bash
python advanced_enhancer.py
```

**Output**: `advanced_enhanced_[filename].mp4`

## Quick Start Guide

### Step-by-Step Process

1. **Setup Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Place Video**: Copy your video file to the project directory

3. **Choose Script**: Select based on your needs:
   - General enhancement → `python horadetect.py`
   - Masked person analysis → `python upperbody_enhancer.py`
   - Face analysis → `python face_enhancer.py`
   - Color improvement → `python color_enhancer.py`
   - Professional analysis → `python advanced_enhancer.py`

4. **Interactive Selection**: All scripts provide:
   - Video file selection menu
   - Click-and-drag ROI selection
   - Processing progress display

5. **Output**: Enhanced video saved automatically

### Script Selection Guide

| Use Case | Script | Best For |
|----------|--------|---------|
| General forensic analysis | `horadetect.py` | Any suspicious activity |
| Masked person identification | `upperbody_enhancer.py` | Upper body features |
| Face detail enhancement | `face_enhancer.py` | Facial features |
| Shadow/lighting issues | `color_enhancer.py` | Poor lighting conditions |
| Maximum detail extraction | `advanced_enhancer.py` | Professional analysis |

## Common Workflow

1. **Select Video** from available files
2. **Interactive ROI Selection**: Click and drag to select area
3. **Processing**: Script applies enhancements
4. **Output**: Enhanced video saved with descriptive filename

## Technical Notes

- **Supported Formats**: MP4, AVI, MOV, MKV
- **Output Format**: MP4 (H.264)
- **Processing**: Frame-by-frame enhancement
- **Memory**: Processes one frame at a time for efficiency
- **Quality**: Uses high-quality interpolation methods

## Troubleshooting

**No video files found**: Ensure video files are in the same directory as scripts

**OpenCV window issues**: Make sure you have GUI support for interactive selection

**Memory issues**: Use smaller videos or reduce zoom factors

**Quality issues**: Try different scripts - `advanced_enhancer.py` for best quality

## Output Files

All scripts create enhanced videos with descriptive prefixes:
- `enhanced_roi_*` - General ROI enhancement
- `upperbody_enhanced_*` - Upper body analysis
- `face_enhanced_*` - Facial enhancement
- `color_enhanced_*` - Color enhancement
- `advanced_enhanced_*` - Advanced processing