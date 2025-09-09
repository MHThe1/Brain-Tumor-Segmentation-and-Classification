"""
Continue Enhanced Attention U-Net Training with Memory Optimization
BRACU CSE428 Academic Project
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.losses import BinaryCrossentropy
import logging

# Memory optimization for GTX 1660 SUPER
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU memory growth enabled for: {gpus[0]}")
    except RuntimeError as e:
        print(f"Memory growth setup failed: {e}")

# Import project modules
from config import *
from utils import *
from data_loader import BrainTumorDataLoader, load_test_data
from models import AttentionUNet, AttentionLoss
from train_attention_unet import EnhancedAttentionUNetTrainer

def train_single_attention_model(model_name, use_self_attention=False, use_channel_attention=True, 
                                use_deep_supervision=False, attention_type='channel'):
    """
    Train a single attention model with optimized settings
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Config: self_attention={use_self_attention}, channel_attention={use_channel_attention}")
    print(f"         deep_supervision={use_deep_supervision}, attention_type={attention_type}")
    print(f"{'='*60}")
    
    # Initialize trainer
    trainer = EnhancedAttentionUNetTrainer()
    
    try:
        model, history, results = trainer.train_attention_unet(
            model_name=model_name,
            use_self_attention=use_self_attention,
            use_channel_attention=use_channel_attention,
            use_deep_supervision=use_deep_supervision,
            attention_type=attention_type
        )
        
        print(f"✅ {model_name} completed successfully!")
        print(f"   Test Dice: {results['test_dice_coefficient']:.4f}")
        print(f"   Test IoU: {results['test_iou_coefficient']:.4f}")
        
        return {
            'status': 'completed',
            'results': results,
            'model_name': model_name
        }
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'model_name': model_name
        }

def main():
    """
    Continue training with memory-optimized configurations
    """
    print("="*80)
    print("CONTINUING ENHANCED ATTENTION U-NET TRAINING")
    print("Memory-Optimized for GTX 1660 SUPER")
    print("="*80)
    
    # Training configurations (memory optimized)
    experiments = [
        {
            'name': 'Enhanced_Attention_UNet_Channel_Only',
            'use_self_attention': False,
            'use_channel_attention': True,
            'use_deep_supervision': False,
            'attention_type': 'channel'
        },
        {
            'name': 'Enhanced_Attention_UNet_Self_Attention_Small',
            'use_self_attention': True,
            'use_channel_attention': False,
            'use_deep_supervision': False,
            'attention_type': 'spatial'
        },
        {
            'name': 'Enhanced_Attention_UNet_Combined_Small',
            'use_self_attention': True,
            'use_channel_attention': True,
            'use_deep_supervision': False,
            'attention_type': 'both'
        }
    ]
    
    results_summary = {}
    
    for exp_config in experiments:
        result = train_single_attention_model(
            model_name=exp_config['name'],
            use_self_attention=exp_config['use_self_attention'],
            use_channel_attention=exp_config['use_channel_attention'],
            use_deep_supervision=exp_config['use_deep_supervision'],
            attention_type=exp_config['attention_type']
        )
        
        results_summary[exp_config['name']] = result
        
        # Clear memory between experiments
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
    
    # Save results
    with open(os.path.join(RUN_OUTPUT_DIR, "results", "continued_training_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=4, default=str)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("CONTINUED TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for name, result in results_summary.items():
        if result['status'] == 'completed':
            dice = result['results']['test_dice_coefficient']
            iou = result['results']['test_iou_coefficient']
            print(f"✅ {name}: Dice={dice:.4f}, IoU={iou:.4f}")
        else:
            print(f"❌ {name}: {result['error']}")
    
    print(f"\nResults saved to: {RUN_OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
