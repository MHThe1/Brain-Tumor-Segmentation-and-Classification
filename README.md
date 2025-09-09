# Brain Tumor Segmentation and Classification
## BRACU CSE428 Academic Project

This project implements advanced deep learning architectures for brain tumor segmentation and classification using the BRISC2025 dataset. **Successfully tested and optimized for GPU acceleration with impressive results!**

## 🎯 **Project Status: FULLY FUNCTIONAL** ✅

- ✅ **Training Pipeline**: Working with 96.98% validation accuracy
- ✅ **Inference Pipeline**: Fast GPU-accelerated predictions
- ✅ **GPU Support**: Optimized for NVIDIA GTX 1660 SUPER
- ✅ **Memory Optimized**: Stable training on 16GB RAM systems
- ✅ **Modern Format**: Using recommended .keras model format

## Project Overview

### Tasks Implemented
1. **U-Net Architecture** - Standard U-Net for brain tumor segmentation
2. **Classification Head** - Tumor type classification (Glioma, Meningioma, No Tumor, Pituitary)
3. **Attention U-Net** - Enhanced U-Net with attention gates
4. **Multi-Task Learning** - Joint segmentation and classification
5. **Inference Pipeline** - Single image analysis and visualization

### Bonus Tasks
1. **Hyperparameter Optimization** - Learning rate, optimizer, and architecture tuning
2. **Classifier Comparison** - Multiple classifier architectures (CNN, ResNet50, EfficientNet, DenseNet)
3. **Performance Analysis** - Comprehensive evaluation and comparison

## Dataset

**BRISC2025 Brain Tumor Dataset**
- **Classification Task**: 4 classes (Glioma, Meningioma, No Tumor, Pituitary)
- **Segmentation Task**: Binary tumor segmentation masks
- **Training**: 5,000+ images with masks
- **Testing**: 1,000+ images with masks

## Project Structure

```
Project/
├── config.py                          # Configuration settings (optimized)
├── utils.py                           # Utility functions
├── data_loader.py                     # Data loading and preprocessing (fixed)
├── models.py                          # Model architectures
├── train.py                           # Training scripts (optimized)
├── inference.py                       # Inference pipeline
├── run_training.py                    # 🚀 Interactive training script
├── run_training_optimized.py          # 🎯 Memory-optimized training script
├── run_inference.py                   # 🔍 Inference script with CLI (fixed)
├── run_jupyter.py                     # 📓 Jupyter launcher script
├── hyperparameter_optimization.py     # Bonus: Hyperparameter tuning
├── classifier_comparison.py           # Bonus: Classifier comparison
├── brain_tumor_analysis.ipynb         # Main Jupyter notebook
├── requirements.txt                   # Dependencies (GPU-enabled, updated)
├── README.md                          # This file
└── outputs/                           # Generated outputs
    └── run_YYYYMMDD_HHMMSS/          # Timestamped run outputs
        ├── models/                    # Trained models (.keras format)
        ├── logs/                      # Training logs
        ├── plots/                     # Visualizations
        └── results/                   # Results and reports
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv_linux
   source venv_linux/bin/activate  # On Linux/Mac
   # On Windows: venv_linux\Scripts\activate
   ```

3. **Install dependencies with GPU support**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: The requirements.txt includes `tensorflow[and-cuda]==2.20.0` for GPU acceleration. Make sure you have:
   - NVIDIA GPU with CUDA support
   - At least 4GB free disk space for CUDA libraries
   - Compatible CUDA drivers installed

4. **Download dataset**
   - Download BRISC2025 dataset from Kaggle
   - Extract to `brisc2025/` directory

## System Requirements

### **Hardware Requirements:**
- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum: 4GB VRAM (GTX 1660, RTX 2060, etc.)
  - Recommended: 8GB+ VRAM for faster training
  - **Tested on**: GTX 1660 SUPER (6GB VRAM) ✅
- **RAM**: 8GB+ system RAM (optimized for 16GB)
- **Storage**: 10GB+ free space for dataset and models

### **Software Requirements:**
- **Python**: 3.8+ (tested with Python 3.12) ✅
- **CUDA**: 12.0+ (automatically installed with tensorflow[and-cuda])
- **Operating System**: Linux, Windows, or macOS

### **GPU Support:**
- ✅ **Automatic GPU detection** and utilization
- ✅ **CUDA acceleration** for 5-10x faster training
- ✅ **Memory management** with automatic growth
- ✅ **Fallback to CPU** if GPU not available

### **Verification:**
Run this to check your setup:
```bash
source venv_linux/bin/activate
python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')))"
```

## 🚀 **Quick Start (Recommended)**

### **1. Memory-Optimized Training (Best for your hardware):**
```bash
source venv_linux/bin/activate
python run_training_optimized.py
```
**Features:**
- ✅ **Optimized for Ryzen 5 3500X + 16GB RAM + GTX 1660 SUPER**
- ✅ **Batch size 16** (reduced from 32 for stability)
- ✅ **50 epochs max** (reduced from 100 for faster training)
- ✅ **Early stopping** (10 epochs patience)
- ✅ **Modern .keras format** (recommended by TensorFlow)
- ✅ **Real-time progress** monitoring

### **2. Interactive Training with Progress Monitoring:**
```bash
source venv_linux/bin/activate
python run_training.py
```
This will show you a menu with options:
- Quick CNN training (5-10 minutes)
- ResNet50 training (15-30 minutes)
- U-Net segmentation training (20-40 minutes)
- Full training pipeline (1-2 hours)

### **3. Run Inference on Images:**
```bash
# Single image inference
python run_inference.py --image path/to/brain_mri.jpg

# Batch inference on a folder
python run_inference.py --batch path/to/images/

# Use different model
python run_inference.py --image path/to/brain_mri.jpg --model resnet50
```

### **4. Launch Jupyter Notebook:**
```bash
python run_jupyter.py
```

## 📊 **Available Models**

### **Classification Models:**
- **Simple CNN**: Fast training, good baseline (5-10 min) - **96.98% accuracy achieved!** ✅
- **ResNet50**: Pre-trained, high accuracy (15-30 min)
- **EfficientNet-B0**: State-of-the-art efficiency (15-30 min)
- **DenseNet121**: Memory efficient, good accuracy (15-30 min)

### **Segmentation Models:**
- **U-Net**: Standard encoder-decoder architecture
- **Attention U-Net**: Enhanced with attention gates
- **Multi-Task U-Net**: Joint segmentation and classification

## 🎯 **Recent Training Results**

### **Simple CNN Classification (Latest Run):**
- **Training Time**: 26.4 minutes
- **Best Validation Accuracy**: **96.98%** (Epoch 47)
- **Final Training Accuracy**: 95.25%
- **Model Format**: .keras (modern format)
- **GPU Utilization**: GTX 1660 SUPER with CUDA acceleration
- **Memory Usage**: Stable with batch size 16

### **Inference Performance:**
- **Glioma Detection**: 99.62% confidence ✅
- **Meningioma Detection**: 85.46% confidence ✅
- **No Tumor Detection**: 99.77% confidence ✅
- **Batch Processing**: 10 images processed successfully ✅

## 🔧 **Preprocessing Pipeline**

### **Image Preprocessing:**
- **Resizing**: All images resized to 256×256 pixels
- **Normalization**: Pixel values normalized to [0, 1] range
- **Color Space**: BGR → RGB conversion for OpenCV compatibility
- **Data Type**: Converted to float32 for TensorFlow

### **Data Augmentation (Training Only):**
- **Rotation**: ±20° random rotation
- **Translation**: ±10% horizontal/vertical shift
- **Horizontal Flip**: 50% chance
- **Zoom**: ±10% zoom range
- **Synchronized**: Image and mask augmented together for segmentation

### **Dataset Statistics:**
- **Classification**: 5,000 training images (4 classes)
- **Segmentation**: 3,933 image-mask pairs
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Train/Val Split**: 80/20 with stratification

## 🔧 **Advanced Usage**

### **Direct Training Script:**
```bash
python train.py
```

### **Programmatic Training:**
```python
from train import BrainTumorTrainer

trainer = BrainTumorTrainer()
# Train U-Net
model, history, results = trainer.train_unet_segmentation()

# Train classification
model, history, results = trainer.train_classification("resnet50")
```

### **Hyperparameter Optimization (Bonus):**
```bash
python hyperparameter_optimization.py
```

### **Classifier Comparison (Bonus):**
```bash
python classifier_comparison.py
```

## Model Architectures

### 1. U-Net
- **Purpose**: Brain tumor segmentation
- **Architecture**: Encoder-decoder with skip connections
- **Input**: 256x256x3 RGB images
- **Output**: 256x256x1 binary segmentation mask

### 2. Attention U-Net
- **Purpose**: Enhanced segmentation with attention gates
- **Architecture**: U-Net with attention mechanisms
- **Benefits**: Better focus on relevant features

### 3. Multi-Task U-Net
- **Purpose**: Joint segmentation and classification
- **Architecture**: U-Net encoder with dual heads
- **Outputs**: Segmentation mask + tumor classification

### 4. Classifier Architectures
- **Simple CNN**: Custom convolutional network
- **ResNet50**: Pre-trained ResNet with custom head
- **EfficientNet**: Pre-trained EfficientNet with custom head
- **DenseNet**: Pre-trained DenseNet with custom head

## 🛠️ **Recent Fixes and Optimizations**

### **Issues Resolved:**
1. ✅ **CPU Overload**: Optimized batch size and epochs for 16GB RAM systems
2. ✅ **Test Data Loading**: Fixed subdirectory structure handling
3. ✅ **Math Domain Error**: Safe evaluation without dummy labels
4. ✅ **Model Format**: Switched to recommended .keras format
5. ✅ **Import Errors**: Fixed missing imports in inference script
6. ✅ **GPU Memory**: Optimized memory usage for GTX 1660 SUPER

### **Performance Improvements:**
- **Training Speed**: 26.4 minutes for 50 epochs (vs. previous crashes)
- **Memory Usage**: Stable with batch size 16
- **GPU Utilization**: Full CUDA acceleration
- **Model Accuracy**: 96.98% validation accuracy achieved

## Academic Features

### Comprehensive Logging
- **Timestamped outputs** for all experiments
- **Detailed training logs** with metrics
- **Model summaries** and architecture details
- **Performance visualizations** and plots

### Reproducible Results
- **Fixed random seeds** for reproducibility
- **Configuration files** for all experiments
- **Detailed experiment reports** in JSON and Markdown
- **Model checkpoints** for best performance

### Performance Tracking
- **Training history plots** (loss, accuracy, metrics)
- **Confusion matrices** for classification
- **Segmentation visualizations** with overlays
- **Comparative analysis** between models

## Results and Outputs

All experiments generate timestamped outputs in `outputs/run_YYYYMMDD_HHMMSS/`:

### Models
- `*_best.keras`: Best model during training (modern format)
- `*_final.keras`: Final model after training
- `*_summary.txt`: Model architecture summary

### Logs
- `training.log`: Detailed training logs
- `*_training.csv`: Training metrics in CSV format

### Plots
- `*_training_history.png`: Training curves
- `*_confusion_matrix.png`: Classification confusion matrix
- `sample_*.png`: Sample predictions and visualizations

### Results
- `*_results.json`: Detailed experiment results
- `*_report.md`: Comprehensive experiment reports
- `experiments_summary.json`: Overall project summary

## Key Metrics

### Segmentation
- **Dice Coefficient**: Overlap between predicted and ground truth masks
- **IoU (Intersection over Union)**: Segmentation accuracy metric

### Classification
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall

## Academic Report Features

### Experiment Documentation
- **Configuration tracking** for all hyperparameters
- **Performance comparison** between architectures
- **Statistical analysis** of results
- **Visualization** of key findings

### Reproducibility
- **Complete code** with detailed comments
- **Environment setup** instructions
- **Dataset preprocessing** documentation
- **Model architecture** specifications

### Analysis and Insights
- **Performance analysis** across different models
- **Hyperparameter sensitivity** analysis
- **Architecture comparison** with trade-offs
- **Recommendations** for future work

## Bonus Tasks Completed

### 1. Hyperparameter Optimization
- **Learning rate tuning** for optimal convergence
- **Optimizer comparison** (Adam, SGD, RMSprop)
- **Architecture parameter** optimization
- **Multi-task loss weight** optimization

### 2. Classifier Architecture Comparison
- **Multiple architectures** tested and compared
- **Performance vs. complexity** analysis
- **Convergence speed** comparison
- **Per-class performance** analysis

### 3. Joint vs. Separate Training Analysis
- **Multi-task learning** implementation
- **Performance comparison** with separate training
- **Loss weight optimization** for balanced learning

## Future Enhancements

### Potential Extensions
1. **EfficientDet Integration** - Advanced decoder architecture
2. **Data Augmentation** - Advanced augmentation techniques
3. **Ensemble Methods** - Combining multiple models
4. **Transfer Learning** - Pre-trained medical imaging models
5. **3D Segmentation** - Volumetric brain tumor analysis

### Research Directions
1. **Attention Mechanisms** - Advanced attention patterns
2. **Loss Function Design** - Custom loss functions for medical imaging
3. **Uncertainty Quantification** - Model confidence estimation
4. **Federated Learning** - Distributed training across institutions

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{brain_tumor_segmentation_2025,
  title={Brain Tumor Segmentation and Classification using Deep Learning},
  author={Md. Mehedi Hasan Tanvir},
  year={2025},
  institution={BRAC University, CSE428},
  note={Academic Project - Successfully implemented with 96.98% accuracy}
}
```

## License

This project is for academic purposes. Please ensure proper attribution when using the code.

## Contact

For questions or collaboration, please contact:
- **Email**: mehedi.hasan.tanvir1@g.bracu.ac.bd
- **Institution**: BRAC University, Department of Computer Science and Engineering
- **Course**: CSE428 - Machine Learning

---

## 🎉 **Project Success Summary**

This project has been **successfully implemented and tested** with the following achievements:

- ✅ **96.98% validation accuracy** on brain tumor classification
- ✅ **GPU acceleration** working perfectly with GTX 1660 SUPER
- ✅ **Memory optimization** for 16GB RAM systems
- ✅ **Modern model format** (.keras) implementation
- ✅ **Complete inference pipeline** with batch processing
- ✅ **All major issues resolved** and system stable

**Note**: This project is designed for academic research and educational purposes. The models and results should be validated with proper medical expertise before any clinical applications.