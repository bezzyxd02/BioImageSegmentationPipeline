# AI-Powered Biomedical Segmentation and Tracking Pipeline

This project is a fully automated deep learning pipeline for segmentation, tracking, and morphometric analysis of complex biomedical signals and microscopy image sequences. It integrates classical image processing, deep learning, and hybrid labeling strategies to produce high-quality cell segmentation and tracking outputs.

The system is designed for high-performance analysis of time-lapse microscopy data, particularly in epithelial cell studies (e.g., MDCK cell cultures).

---

## Key Features

- Fully automated preprocessing of raw microscopy images
- Hybrid labeling using Cellpose and Ilastik probability maps
- Deep learning segmentation using a custom U-Net architecture
- Patch-based training strategy for high-resolution images
- Mixed-precision training for GPU efficiency
- Post-processing for morphological refinement
- Multi-object cell tracking across time frames
- Automated morphometric feature extraction
- Export of results (masks, overlays, CSV analytics)
- GUI-based pipeline execution (Tkinter interface)

---

## Pipeline Overview

### 1. Data Loading
- Loads multi-frame TIFF microscopy datasets
- Supports batch processing of multiple experiments

### 2. Preprocessing
- Percentile normalization
- Hot pixel removal
- Background subtraction
- Gaussian filtering
- Non-local means denoising
- CLAHE contrast enhancement

### 3. Hybrid Label Generation
- Cellpose-based segmentation
- Optional Ilastik probability map integration
- Weighted fusion of multiple labeling sources

### 4. Model Architecture
A custom U-Net is used:

- Encoder-decoder structure
- Batch normalization + dropout regularization
- Skip connections for spatial precision
- Fully convolutional design for dense prediction

### 5. Training Strategy
- Patch-based dataset generation
- BCE + Dice loss function
- AdamW optimizer
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping mechanism
- Gradient clipping
- TensorBoard logging support
- Mixed precision training (AMP)

### 6. Inference
- Frame-by-frame segmentation
- Post-processing with morphological filtering
- Hole filling and noise removal

### 7. Tracking
- Centroid extraction per object
- Hungarian algorithm (linear sum assignment)
- Distance-based temporal association

### 8. Feature Extraction
For each detected object:
- Area
- Perimeter
- Circularity
- Eccentricity
- Solidity
- Aspect ratio
- Intensity statistics
- Centroid position

### 9. Output Export
- Segmentation masks (TIFF stack)
- Overlay visualizations (PNG)
- Morphometric CSV tables
- Tracking trajectories CSV

---

## Model Architecture

The segmentation model is based on a custom U-Net:

- Input: grayscale microscopy images
- Encoder: 3 downsampling convolution blocks
- Bottleneck representation
- Decoder: transpose convolutions with skip connections
- Output: binary segmentation map

Loss function:
- Combination of Binary Cross Entropy and Dice Loss

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- SciPy
- scikit-image
- tifffile
- pandas
- TensorBoard
- Cellpose
- Ilastik (probability maps)
- Tkinter (GUI)

---

## System Requirements

- CUDA-compatible GPU (recommended)
- Python 3.9+
- PyTorch with CUDA support

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
