#!/usr/bin/env python3
"""
Memory-Optimized Brain Tumor Training Script
BRACU CSE428 Academic Project
Optimized for Ryzen 5 3500X + 16GB RAM + GTX 1660 SUPER
"""

import os
import sys
import time
import tensorflow as tf
from train import BrainTumorTrainer

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def check_system_resources():
    """Check system resources and optimize settings"""
    print("\n🔍 Checking System Resources...")
    print("-" * 30)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0].name}")
        try:
            memory_info = tf.config.experimental.get_memory_info(gpus[0].name)
            print(f"✅ GPU memory: {memory_info['current'] / 1024**3:.1f}GB / {memory_info['peak'] / 1024**3:.1f}GB")
        except:
            print("✅ GPU memory info not available")
    else:
        print("❌ No GPU detected, using CPU")
    
    # Memory optimization settings
    print("\n⚙️  Memory Optimization Settings:")
    print("   - Batch size: 16 (reduced from 32)")
    print("   - Epochs: 50 (reduced from 100)")
    print("   - Early stopping: 10 epochs (reduced from 15)")
    print("   - Model format: .keras (newer format)")
    
    return len(gpus) > 0

def run_optimized_training(model_type="simple_cnn"):
    """Run training with memory optimizations"""
    print_header(f"Memory-Optimized {model_type.upper()} Training")
    
    # Set memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("✅ GPU memory growth enabled")
        except:
            print("⚠️  Could not set GPU memory growth")
    
    # Initialize trainer
    trainer = BrainTumorTrainer()
    
    print(f"📊 Starting {model_type} training with optimizations...")
    print("📈 Training progress will be shown below:")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Train the model
        model, history, results = trainer.train_classification(model_type)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"⏱️  Training time: {training_time/60:.1f} minutes")
        print(f"🎯 Final test accuracy: {results['test_accuracy']:.4f}")
        print(f"📁 Results saved to: {trainer.output_dir}")
        
        return model, history, results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None, None, None

def main():
    """Main function"""
    print_header("Memory-Optimized Brain Tumor Training")
    print("BRACU CSE428 Academic Project")
    print("Optimized for: Ryzen 5 3500X + 16GB RAM + GTX 1660 SUPER")
    
    # Check system resources
    gpu_available = check_system_resources()
    
    # Ask user what to run
    print("\n🎯 What would you like to run?")
    print("1. Quick CNN training (3-5 minutes)")
    print("2. ResNet50 training (8-15 minutes)")
    print("3. U-Net segmentation training (10-20 minutes)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_optimized_training("simple_cnn")
    elif choice == "2":
        run_optimized_training("resnet50")
    elif choice == "3":
        print_header("U-Net Segmentation Training")
        trainer = BrainTumorTrainer()
        print("📊 Starting U-Net segmentation training...")
        model, history, results = trainer.train_unet_segmentation()
        print(f"✅ U-Net training completed! Dice: {results['test_dice_coefficient']:.4f}")
    elif choice == "4":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please run the script again.")
        return
    
    print("\n🎉 Training session completed!")
    print("📁 Check the outputs/ directory for results, plots, and models.")

if __name__ == "__main__":
    main()
