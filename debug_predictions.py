#!/usr/bin/env python3
"""
Debug script to investigate prediction issues
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import project modules
from config import *
from data_loader import BrainTumorDataLoader, load_test_data
from utils import dice_coefficient, iou_coefficient

def debug_predictions():
    """Debug the prediction issues"""
    
    print("🔍 DEBUGGING PREDICTION ISSUES")
    print("=" * 50)
    
    # Find the latest trained model
    model_dir = None
    latest_run = None
    
    if os.path.exists(OUTPUT_ROOT):
        runs = [d for d in os.listdir(OUTPUT_ROOT) if d.startswith('run_')]
        if runs:
            for run in sorted(runs, reverse=True):
                potential_model_dir = os.path.join(OUTPUT_ROOT, run, 'models')
                model_file = os.path.join(potential_model_dir, 'unet_segmentation_best.keras')
                if os.path.exists(model_file):
                    latest_run = run
                    model_dir = potential_model_dir
                    break
    
    if not model_dir:
        print("❌ No trained model found!")
        return
    
    model_file = os.path.join(model_dir, 'unet_segmentation_best.keras')
    print(f"📦 Loading model from: {model_file}")
    
    # Load the model
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'iou_coefficient': iou_coefficient
    }
    
    try:
        model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load a small sample of test data
    print("\n📊 Loading test data sample...")
    try:
        dummy_loader = BrainTumorDataLoader()
        test_images, test_masks = load_test_data(
            dummy_loader,
            SEGMENTATION_TEST_IMAGES, 
            SEGMENTATION_TEST_MASKS
        )
        
        # Take only first 5 images for debugging
        test_images = test_images[:5]
        test_masks = test_masks[:5]
        
        print(f"✅ Loaded {len(test_images)} test images for debugging")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Images and masks are already normalized in load_test_data
    test_images = test_images.astype(np.float32)
    test_masks = test_masks.astype(np.float32)
    
    print(f"📐 Image shape: {test_images.shape}")
    print(f"📐 Mask shape: {test_masks.shape}")
    print(f"📊 Image range: {test_images.min():.4f} - {test_images.max():.4f}")
    print(f"📊 Mask range: {test_masks.min():.4f} - {test_masks.max():.4f}")
    
    # Get predictions
    print("\n🔍 Getting predictions...")
    predictions = model.predict(test_images, batch_size=1, verbose=1)
    
    print(f"📐 Prediction shape: {predictions.shape}")
    print(f"📊 Prediction range: {predictions.min():.8f} - {predictions.max():.8f}")
    print(f"📊 Prediction mean: {predictions.mean():.8f}")
    print(f"📊 Prediction std: {predictions.std():.8f}")
    
    # Check different thresholds
    print("\n🎯 Testing different thresholds:")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(np.float32)
        non_zero_pixels = np.sum(pred_binary > 0)
        total_pixels = pred_binary.size
        print(f"   Threshold {threshold}: {non_zero_pixels}/{total_pixels} pixels ({non_zero_pixels/total_pixels*100:.2f}%)")
    
    # Calculate metrics with different thresholds
    print("\n📊 Metrics with different thresholds:")
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(np.float32)
        dice_scores = []
        iou_scores = []
        
        for i in range(len(test_images)):
            dice = dice_coefficient(test_masks[i], pred_binary[i]).numpy()
            iou = iou_coefficient(test_masks[i], pred_binary[i]).numpy()
            dice_scores.append(dice)
            iou_scores.append(iou)
        
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        print(f"   Threshold {threshold}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")
    
    # Visualize first prediction
    print("\n🖼️  Visualizing first prediction...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(test_images[0])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(test_masks[0].squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Raw prediction
    axes[2].imshow(predictions[0].squeeze(), cmap='gray')
    axes[2].set_title(f'Raw Prediction\nRange: {predictions[0].min():.6f} - {predictions[0].max():.6f}')
    axes[2].axis('off')
    
    # Binary prediction (threshold 0.5)
    pred_binary = (predictions[0] > 0.5).astype(np.float32)
    axes[3].imshow(pred_binary.squeeze(), cmap='gray')
    axes[3].set_title(f'Binary Prediction (0.5)\nNon-zero: {np.sum(pred_binary > 0)} pixels')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save the debug visualization
    debug_file = os.path.join(OUTPUT_ROOT, latest_run, 'debug_predictions.png')
    plt.savefig(debug_file, dpi=150, bbox_inches='tight')
    print(f"💾 Debug visualization saved to: {debug_file}")
    
    plt.show()

if __name__ == "__main__":
    debug_predictions()
