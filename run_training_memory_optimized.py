#!/usr/bin/env python3
"""
Memory-optimized training script for Brain Tumor Segmentation
Optimized for GTX 1660 SUPER (6GB VRAM)
"""

import os
import sys
import tensorflow as tf
from train import BrainTumorTrainer
from config import *

def setup_memory_optimization():
    """Configure TensorFlow for memory optimization"""
    print("🔧 Setting up memory optimization for GTX 1660 SUPER...")
    
    # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled for GPU")
        except RuntimeError as e:
            print(f"⚠️  Memory growth setup failed: {e}")
    
    # Note: Mixed precision disabled due to type compatibility issues
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ Using float32 precision (mixed precision disabled for compatibility)")

def main():
    """Main training function with memory optimization"""
    print("=" * 60)
    print("🚀 Memory-Optimized Brain Tumor Segmentation Training")
    print("=" * 60)
    print("🎯 Target: GTX 1660 SUPER (6GB VRAM)")
    print("📊 Batch Size: 8 (reduced for memory)")
    print("🏗️  Model: U-Net with 32 filters (memory optimized)")
    print("=" * 60)
    
    # Setup memory optimization
    setup_memory_optimization()
    
    # Create trainer
    trainer = BrainTumorTrainer()
    
    try:
        print("\n🚀 Starting memory-optimized U-Net segmentation training...")
        print("📝 This may take 30-45 minutes with reduced batch size...")
        
        # Train U-Net segmentation
        model, history, results = trainer.train_unet_segmentation()
        
        print("\n✅ Memory-optimized U-Net training completed successfully!")
        print(f"📁 Results saved to: {RUN_OUTPUT_DIR}")
        
        # Print key metrics
        if 'dice_coefficient' in history.history:
            best_dice = max(history.history['dice_coefficient'])
            best_iou = max(history.history['iou_coefficient'])
            print(f"🎯 Best Dice Coefficient: {best_dice:.4f}")
            print(f"🎯 Best IoU Coefficient: {best_iou:.4f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
