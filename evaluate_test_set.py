#!/usr/bin/env python3
"""
Comprehensive test set evaluation for U-Net segmentation model
Evaluates all 860 test images and provides detailed metrics
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from config import *
from data_loader import load_test_data
from utils import dice_coefficient, iou_coefficient

def create_visualization(test_images, test_masks, predictions, dice_scores, iou_scores, output_dir, latest_run):
    """Create comprehensive visualization of test results"""
    
    print("\n🖼️  Creating comprehensive visualization...")
    
    # Find best and worst performing cases
    best_indices = np.argsort(dice_scores)[-6:]  # Top 6
    worst_indices = np.argsort(dice_scores)[:6]  # Bottom 6
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    fig.suptitle('U-Net Segmentation Test Results - Best and Worst Cases', fontsize=16, fontweight='bold')
    
    # Plot best cases (top 2 rows)
    for i, idx in enumerate(best_indices):
        row = i // 3
        col = i % 3
        
        # Original image
        axes[row, col].imshow(test_images[idx])
        axes[row, col].set_title(f'Best #{i+1}\nDice: {dice_scores[idx]:.3f}, IoU: {iou_scores[idx]:.3f}', 
                                fontsize=10, fontweight='bold', color='green')
        axes[row, col].axis('off')
        
        # Ground truth mask
        axes[row, col+3].imshow(test_masks[idx].squeeze(), cmap='gray')
        axes[row, col+3].set_title('Ground Truth', fontsize=10)
        axes[row, col+3].axis('off')
    
    # Plot worst cases (bottom 2 rows)
    for i, idx in enumerate(worst_indices):
        row = (i // 3) + 2
        col = i % 3
        
        # Original image
        axes[row, col].imshow(test_images[idx])
        axes[row, col].set_title(f'Worst #{i+1}\nDice: {dice_scores[idx]:.3f}, IoU: {iou_scores[idx]:.3f}', 
                                fontsize=10, fontweight='bold', color='red')
        axes[row, col].axis('off')
        
        # Ground truth mask
        axes[row, col+3].imshow(test_masks[idx].squeeze(), cmap='gray')
        axes[row, col+3].set_title('Ground Truth', fontsize=10)
        axes[row, col+3].axis('off')
    
    # Add predictions for best cases
    for i, idx in enumerate(best_indices):
        row = i // 3
        col = (i % 3) + 3
        
        pred_binary = (predictions[idx] > 0.5).astype(np.float32)
        axes[row, col].imshow(pred_binary.squeeze(), cmap='gray')
        axes[row, col].set_title('Prediction', fontsize=10)
        axes[row, col].axis('off')
    
    # Add predictions for worst cases
    for i, idx in enumerate(worst_indices):
        row = (i // 3) + 2
        col = (i % 3) + 3
        
        pred_binary = (predictions[idx] > 0.5).astype(np.float32)
        axes[row, col].imshow(pred_binary.squeeze(), cmap='gray')
        axes[row, col].set_title('Prediction', fontsize=10)
        axes[row, col].axis('off')
    
    # Add column labels
    axes[0, 0].text(-0.1, 0.5, 'BEST CASES', transform=axes[0, 0].transAxes, 
                    rotation=90, va='center', ha='center', fontsize=12, fontweight='bold', color='green')
    axes[2, 0].text(-0.1, 0.5, 'WORST CASES', transform=axes[2, 0].transAxes, 
                    rotation=90, va='center', ha='center', fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Save the comprehensive visualization
    viz_file = os.path.join(output_dir, latest_run, 'comprehensive_test_visualization.png')
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"💾 Comprehensive visualization saved to: {viz_file}")
    
    # Create performance distribution plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Dice score distribution
    ax1.hist(dice_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(dice_scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(dice_scores):.3f}')
    ax1.set_xlabel('Dice Coefficient')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Dice Coefficient Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IoU score distribution
    ax2.hist(iou_scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(np.mean(iou_scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(iou_scores):.3f}')
    ax2.set_xlabel('IoU Coefficient')
    ax2.set_ylabel('Frequency')
    ax2.set_title('IoU Coefficient Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the distribution plot
    dist_file = os.path.join(output_dir, latest_run, 'test_performance_distribution.png')
    plt.savefig(dist_file, dpi=150, bbox_inches='tight')
    print(f"💾 Performance distribution saved to: {dist_file}")
    
    plt.close('all')

def evaluate_test_set():
    """Evaluate the trained U-Net model on the complete test set"""
    
    print("=" * 80)
    print("🧪 COMPREHENSIVE TEST SET EVALUATION")
    print("=" * 80)
    print(f"📁 Test Images: {SEGMENTATION_TEST_IMAGES}")
    print(f"📁 Test Masks: {SEGMENTATION_TEST_MASKS}")
    print("=" * 80)
    
    # Find the latest trained model
    model_dir = None
    latest_run = None
    
    # Look for the most recent run directory
    if os.path.exists(OUTPUT_ROOT):
        runs = [d for d in os.listdir(OUTPUT_ROOT) if d.startswith('run_')]
        if runs:
            # Find the run with the trained model
            for run in sorted(runs, reverse=True):
                potential_model_dir = os.path.join(OUTPUT_ROOT, run, 'models')
                model_file = os.path.join(potential_model_dir, 'unet_segmentation_best.keras')
                if os.path.exists(model_file):
                    latest_run = run
                    model_dir = potential_model_dir
                    break
    
    if not model_dir or not os.path.exists(model_dir):
        print("❌ No trained model found!")
        return
    
    # Find the best model file
    model_file = os.path.join(model_dir, 'unet_segmentation_best.keras')
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return
    
    print(f"📦 Loading model from: {model_file}")
    
    # Load the model with custom objects
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
    
    # Load test data
    print("\n📊 Loading test dataset...")
    try:
        # Create a dummy data loader for the function
        from data_loader import BrainTumorDataLoader
        dummy_loader = BrainTumorDataLoader()
        test_images, test_masks = load_test_data(
            dummy_loader,
            SEGMENTATION_TEST_IMAGES, 
            SEGMENTATION_TEST_MASKS
        )
        print(f"✅ Loaded {len(test_images)} test images and {len(test_masks)} test masks")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Images and masks are already normalized in load_test_data
    test_images = test_images.astype(np.float32)
    test_masks = test_masks.astype(np.float32)
    
    print(f"📐 Image shape: {test_images.shape}")
    print(f"📐 Mask shape: {test_masks.shape}")
    
    # Evaluate model
    print("\n🔍 Evaluating model on test set...")
    print("⏳ This may take a few minutes...")
    
    # Get predictions
    predictions = model.predict(test_images, batch_size=8, verbose=1)
    
    # Threshold predictions to binary
    predictions_binary = (predictions > 0.5).astype(np.float32)
    
    # Calculate metrics for each image
    dice_scores = []
    iou_scores = []
    
    print("\n📊 Calculating individual metrics...")
    for i in tqdm(range(len(test_images)), desc="Processing"):
        pred = predictions_binary[i]
        true = test_masks[i]
        
        # Calculate Dice coefficient
        dice = dice_coefficient(true, pred).numpy()
        dice_scores.append(dice)
        
        # Calculate IoU coefficient
        iou = iou_coefficient(true, pred).numpy()
        iou_scores.append(iou)
    
    # Convert to numpy arrays
    dice_scores = np.array(dice_scores)
    iou_scores = np.array(iou_scores)
    
    # Calculate overall metrics
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    
    # Calculate additional statistics
    perfect_predictions = np.sum(dice_scores > 0.9)
    good_predictions = np.sum(dice_scores > 0.7)
    poor_predictions = np.sum(dice_scores < 0.3)
    no_detection = np.sum(dice_scores == 0.0)
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SET RESULTS")
    print("=" * 80)
    print(f"📈 Total Test Images: {len(test_images)}")
    print(f"🎯 Mean Dice Coefficient: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"🎯 Mean IoU Coefficient: {mean_iou:.4f} ± {std_iou:.4f}")
    print()
    print("📊 Performance Distribution:")
    print(f"   🏆 Excellent (Dice > 0.9): {perfect_predictions} ({perfect_predictions/len(test_images)*100:.1f}%)")
    print(f"   ✅ Good (Dice > 0.7): {good_predictions} ({good_predictions/len(test_images)*100:.1f}%)")
    print(f"   ⚠️  Poor (Dice < 0.3): {poor_predictions} ({poor_predictions/len(test_images)*100:.1f}%)")
    print(f"   ❌ No Detection (Dice = 0.0): {no_detection} ({no_detection/len(test_images)*100:.1f}%)")
    print()
    print("📈 Score Ranges:")
    print(f"   Dice Coefficient: {np.min(dice_scores):.4f} - {np.max(dice_scores):.4f}")
    print(f"   IoU Coefficient: {np.min(iou_scores):.4f} - {np.max(iou_scores):.4f}")
    print("=" * 80)
    
    # Save detailed results
    results = {
        'model_file': model_file,
        'test_set_size': len(test_images),
        'mean_dice': float(mean_dice),
        'std_dice': float(std_dice),
        'mean_iou': float(mean_iou),
        'std_iou': float(std_iou),
        'perfect_predictions': int(perfect_predictions),
        'good_predictions': int(good_predictions),
        'poor_predictions': int(poor_predictions),
        'no_detection': int(no_detection),
        'min_dice': float(np.min(dice_scores)),
        'max_dice': float(np.max(dice_scores)),
        'min_iou': float(np.min(iou_scores)),
        'max_iou': float(np.max(iou_scores)),
        'individual_dice_scores': dice_scores.tolist(),
        'individual_iou_scores': iou_scores.tolist(),
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Save to results directory
    results_file = os.path.join(OUTPUT_ROOT, latest_run, 'results', 'comprehensive_test_evaluation.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Create comprehensive visualization
    create_visualization(test_images, test_masks, predictions, dice_scores, iou_scores, OUTPUT_ROOT, latest_run)
    
    print("✅ Test evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_test_set()
