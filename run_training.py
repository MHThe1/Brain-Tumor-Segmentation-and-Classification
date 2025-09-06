#!/usr/bin/env python3
"""
Brain Tumor Training Script with Progress Monitoring
BRACU CSE428 Academic Project
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

def check_gpu_status():
    """Check and display GPU status"""
    print("\n🔍 Checking GPU Status...")
    print("-" * 30)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0].name}")
        try:
            memory_info = tf.config.experimental.get_memory_info(gpus[0].name)
            print(f"✅ GPU memory: {memory_info['current'] / 1024**3:.1f}GB / {memory_info['peak'] / 1024**3:.1f}GB")
        except:
            print("✅ GPU memory info not available")
        print(f"✅ CUDA available: {tf.test.is_built_with_cuda()}")
    else:
        print("❌ No GPU detected, using CPU")
    
    print(f"✅ TensorFlow version: {tf.__version__}")

def run_single_training(model_type="simple_cnn"):
    """Run a single model training with progress monitoring"""
    print_header(f"Training {model_type.upper()} Model")
    
    # Initialize trainer
    trainer = BrainTumorTrainer()
    
    print(f"📊 Starting {model_type} training...")
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

def run_quick_test():
    """Run a quick test to verify everything works"""
    print_header("Quick System Test")
    
    # Test GPU
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"✅ Matrix multiplication test: {c.numpy().tolist()}")
    
    # Test data loading
    print("\n📊 Testing data loading...")
    trainer = BrainTumorTrainer()
    try:
        from config import CLASSIFICATION_TRAIN, SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        
        (X_train, y_train), (X_val, y_val) = trainer.data_loader.load_classification_data(
            CLASSIFICATION_TRAIN
        )
        print(f"✅ Classification data loaded: {X_train.shape[0]} training samples")
        
        (X_seg_train, y_seg_train), (X_seg_val, y_seg_val) = trainer.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, 
            SEGMENTATION_TRAIN_MASKS
        )
        print(f"✅ Segmentation data loaded: {X_seg_train.shape[0]} training samples")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    print("✅ All tests passed! System is ready for training.")
    return True

def main():
    """Main function"""
    print_header("Brain Tumor Segmentation and Classification")
    print("BRACU CSE428 Academic Project")
    print("GPU-Accelerated Training Pipeline")
    
    # Check GPU status
    check_gpu_status()
    
    # Run quick test
    if not run_quick_test():
        print("❌ System test failed. Please check your setup.")
        return
    
    # Ask user what to run
    print("\n🎯 What would you like to run?")
    print("1. Quick CNN training (5-10 minutes)")
    print("2. ResNet50 training (15-30 minutes)")
    print("3. U-Net segmentation training (20-40 minutes)")
    print("4. Full training pipeline (1-2 hours)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        run_single_training("simple_cnn")
    elif choice == "2":
        run_single_training("resnet50")
    elif choice == "3":
        print_header("U-Net Segmentation Training")
        trainer = BrainTumorTrainer()
        print("📊 Starting U-Net segmentation training...")
        model, history, results = trainer.train_unet_segmentation()
        print(f"✅ U-Net training completed! Dice: {results['test_dice_coefficient']:.4f}")
    elif choice == "4":
        print_header("Full Training Pipeline")
        trainer = BrainTumorTrainer()
        print("📊 Running all experiments...")
        # This will run all models as defined in train.py main()
        from train import main as train_main
        train_main()
    elif choice == "5":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please run the script again.")
        return
    
    print("\n🎉 Training session completed!")
    print("📁 Check the outputs/ directory for results, plots, and models.")

if __name__ == "__main__":
    main()
