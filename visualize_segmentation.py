#!/usr/bin/env python3
"""
Brain Tumor Segmentation Visualization
BRACU CSE428 Academic Project

This script creates visualizations showing:
- Original test images
- Ground truth masks
- Predicted masks
- Overlay comparisons

Usage: python visualize_segmentation.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import logging

# Define custom functions for model loading
def dice_coefficient(y_true, y_pred):
    """Dice coefficient metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection) / (union + 1e-8)

def iou_coefficient(y_true, y_pred):
    """IoU coefficient metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-8)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_model():
    """Load the most recent U-Net segmentation model"""
    # Find the latest run directory
    output_dirs = glob.glob("outputs/run_*")
    if not output_dirs:
        raise FileNotFoundError("No output directories found")
    
    latest_dir = max(output_dirs, key=os.path.getctime)
    model_path = os.path.join(latest_dir, "models", "unet_segmentation_best.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model with custom objects
    with custom_object_scope({
        'dice_coefficient': dice_coefficient,
        'iou_coefficient': iou_coefficient
    }):
        model = load_model(model_path)
    
    return model, latest_dir

def load_test_data():
    """Load test images and masks for segmentation"""
    test_images_dir = "brisc2025/segmentation_task/test/images"
    test_masks_dir = "brisc2025/segmentation_task/test/masks"
    
    if not os.path.exists(test_images_dir) or not os.path.exists(test_masks_dir):
        raise FileNotFoundError("Test data directories not found")
    
    # Load test images
    test_images = []
    test_masks = []
    image_files = []
    
    # Get all image files
    image_files_list = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))
    
    for img_path in image_files_list[:6]:  # Load only first 6 images
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (256, 256))
        image_normalized = image_resized.astype(np.float32) / 255.0
        test_images.append(image_normalized)
        
        # Load corresponding mask
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(test_masks_dir, img_name.replace('.jpg', '.png'))
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask, (256, 256))
            mask_normalized = (mask_resized > 128).astype(np.float32)  # Binary mask
            test_masks.append(mask_normalized)
        else:
            logger.warning(f"Mask not found for {img_name}")
            # Create empty mask if not found
            test_masks.append(np.zeros((256, 256), dtype=np.float32))
        
        image_files.append(img_name)
    
    return np.array(test_images), np.array(test_masks), image_files

def predict_masks(model, test_images):
    """Generate predictions for test images"""
    logger.info("Generating predictions...")
    predictions = model.predict(test_images, verbose=1)
    
    # Handle shape - predictions might be (batch, height, width, 1) or (batch, height, width)
    if len(predictions.shape) == 4:
        predictions = np.squeeze(predictions, axis=-1)
    
    # Convert to binary masks (threshold at 0.5)
    binary_predictions = (predictions > 0.5).astype(np.float32)
    
    return binary_predictions

def create_overlay(image, mask, alpha=0.5):
    """Create overlay of mask on image"""
    # Convert image to uint8
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Create colored mask (red for tumor)
    colored_mask = np.zeros_like(img_uint8)
    colored_mask[:, :, 0] = mask * 255  # Red channel
    
    # Create overlay
    overlay = cv2.addWeighted(img_uint8, 1-alpha, colored_mask, alpha, 0)
    return overlay

def visualize_segmentation_results(model, test_images, test_masks, image_files, output_dir):
    """Create comprehensive visualization of segmentation results"""
    
    # Generate predictions
    predictions = predict_masks(model, test_images)
    
    # Create figure with subplots
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    fig.suptitle('Brain Tumor Segmentation Results\n(Original Image | Ground Truth Mask | Predicted Mask | Overlay Comparison)', 
                 fontsize=16, fontweight='bold')
    
    for i in range(6):
        # Original image
        axes[i, 0].imshow(test_images[i])
        axes[i, 0].set_title(f'Original Image\n{image_files[i]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(test_masks[i], cmap='Reds', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth Mask', fontsize=10)
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(predictions[i], cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title('Predicted Mask', fontsize=10)
        axes[i, 2].axis('off')
        
        # Overlay comparison
        overlay_gt = create_overlay(test_images[i], test_masks[i])
        overlay_pred = create_overlay(test_images[i], predictions[i])
        
        # Create side-by-side overlay
        combined_overlay = np.hstack([overlay_gt, overlay_pred])
        axes[i, 3].imshow(combined_overlay)
        axes[i, 3].set_title('Overlay Comparison\n(Left: GT, Right: Pred)', fontsize=10)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(output_dir, "segmentation_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to: {output_path}")
    
    # Calculate and display metrics
    calculate_metrics(test_masks, predictions, image_files)
    
    return output_path

def calculate_metrics(gt_masks, pred_masks, image_files):
    """Calculate and display segmentation metrics"""
    logger.info("\n" + "="*60)
    logger.info("SEGMENTATION METRICS")
    logger.info("="*60)
    
    dice_scores = []
    iou_scores = []
    
    for i in range(len(gt_masks)):
        # Calculate Dice coefficient
        intersection = np.sum(gt_masks[i] * pred_masks[i])
        union = np.sum(gt_masks[i]) + np.sum(pred_masks[i])
        dice = (2.0 * intersection) / (union + 1e-8)
        dice_scores.append(dice)
        
        # Calculate IoU
        intersection = np.sum(gt_masks[i] * pred_masks[i])
        union = np.sum(gt_masks[i]) + np.sum(pred_masks[i]) - intersection
        iou = intersection / (union + 1e-8)
        iou_scores.append(iou)
        
        logger.info(f"{image_files[i]:<30} | Dice: {dice:.4f} | IoU: {iou:.4f}")
    
    # Overall metrics
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    
    logger.info("-" * 60)
    logger.info(f"{'AVERAGE':<30} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
    logger.info("="*60)

def main():
    """Main function"""
    try:
        logger.info("Starting segmentation visualization...")
        
        # Load the latest trained model
        model, output_dir = load_latest_model()
        
        # Load test data
        test_images, test_masks, image_files = load_test_data()
        logger.info(f"Loaded {len(test_images)} test images")
        
        # Create visualization
        output_path = visualize_segmentation_results(
            model, test_images, test_masks, image_files, output_dir
        )
        
        logger.info(f"\n✅ Visualization completed successfully!")
        logger.info(f"📊 Check the results at: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
