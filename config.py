"""
Configuration file for Brain Tumor Segmentation and Classification Project
BRACU CSE428 Academic Project
"""

import os
from datetime import datetime

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "brisc2025")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")

# Create output directories with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"run_{TIMESTAMP}")

# Dataset paths
CLASSIFICATION_TRAIN = os.path.join(DATA_ROOT, "classification_task", "train")
CLASSIFICATION_TEST = os.path.join(DATA_ROOT, "classification_task", "test")
SEGMENTATION_TRAIN_IMAGES = os.path.join(DATA_ROOT, "segmentation_task", "train", "images")
SEGMENTATION_TRAIN_MASKS = os.path.join(DATA_ROOT, "segmentation_task", "train", "masks")
SEGMENTATION_TEST_IMAGES = os.path.join(DATA_ROOT, "segmentation_task", "test", "images")
SEGMENTATION_TEST_MASKS = os.path.join(DATA_ROOT, "segmentation_task", "test", "masks")

# Model parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
NUM_CLASSES = 4  # glioma, meningioma, no_tumor, pituitary
SEGMENTATION_CLASSES = 2  # background, tumor

# Training parameters
BATCH_SIZE = 8   # Reduced for U-Net memory requirements (GTX 1660 SUPER 6GB)
EPOCHS = 50      # Reduced for faster training
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Model architectures
MODELS = {
    'unet': 'U-Net',
    'attention_unet': 'Attention U-Net',
    'efficientdet_unet': 'EfficientDet U-Net'
}

# Classifier architectures
CLASSIFIERS = {
    'simple_cnn': 'Simple CNN',
    'resnet50': 'ResNet50',
    'efficientnet': 'EfficientNet',
    'densenet': 'DenseNet'
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create necessary directories
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(RUN_OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RUN_OUTPUT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(RUN_OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RUN_OUTPUT_DIR, "results"), exist_ok=True)

print(f"Project initialized with timestamp: {TIMESTAMP}")
print(f"Output directory: {RUN_OUTPUT_DIR}")
