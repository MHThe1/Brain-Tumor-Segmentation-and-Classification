"""
Test script for Enhanced Attention U-Net
BRACU CSE428 Academic Project
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Import project modules
from config import *
from utils import *
from models import AttentionUNet, AttentionLoss

def test_attention_unet_models():
    """
    Test different configurations of the Enhanced Attention U-Net
    """
    print("="*60)
    print("TESTING ENHANCED ATTENTION U-NET MODELS")
    print("="*60)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Basic_Attention_UNet',
            'use_self_attention': False,
            'use_channel_attention': False,
            'use_deep_supervision': False,
            'attention_type': 'spatial'
        },
        {
            'name': 'Spatial_Attention_UNet',
            'use_self_attention': False,
            'use_channel_attention': False,
            'use_deep_supervision': False,
            'attention_type': 'spatial'
        },
        {
            'name': 'Channel_Attention_UNet',
            'use_self_attention': False,
            'use_channel_attention': True,
            'use_deep_supervision': False,
            'attention_type': 'channel'
        },
        {
            'name': 'Self_Attention_UNet',
            'use_self_attention': True,
            'use_channel_attention': False,
            'use_deep_supervision': False,
            'attention_type': 'spatial'
        },
        {
            'name': 'Full_Enhanced_Attention_UNet',
            'use_self_attention': True,
            'use_channel_attention': True,
            'use_deep_supervision': True,
            'attention_type': 'both'
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 40)
        
        try:
            # Create model
            attention_unet = AttentionUNet(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                num_classes=1,
                filters=16,  # Small for testing
                use_self_attention=config['use_self_attention'],
                use_channel_attention=config['use_channel_attention'],
                use_deep_supervision=config['use_deep_supervision'],
                attention_type=config['attention_type']
            )
            
            model = attention_unet.build_model()
            
            # Print model summary
            print(f"Model created successfully!")
            print(f"Number of parameters: {model.count_params():,}")
            
            # Test compilation
            if config['use_deep_supervision']:
                loss_fn = AttentionLoss.deep_supervision_loss()
                metrics = {
                    'main_output': [dice_coefficient, iou_coefficient],
                    'aux_output_4': [dice_coefficient, iou_coefficient],
                    'aux_output_3': [dice_coefficient, iou_coefficient],
                    'aux_output_2': [dice_coefficient, iou_coefficient]
                }
            else:
                loss_fn = BinaryCrossentropy()
                metrics = [dice_coefficient, iou_coefficient]
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=loss_fn,
                metrics=metrics
            )
            
            print("Model compiled successfully!")
            
            # Test forward pass
            test_input = np.random.random((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)).astype(np.float32)
            test_output = model.predict(test_input, verbose=0)
            
            if config['use_deep_supervision']:
                print(f"Output shapes: {[output.shape for output in test_output]}")
                print(f"Main output shape: {test_output[0].shape}")
            else:
                print(f"Output shape: {test_output.shape}")
            
            # Test attention map extraction
            try:
                attention_maps = attention_unet.get_attention_maps(model, test_input)
                print(f"Attention maps extracted: {len(attention_maps)} maps")
            except Exception as e:
                print(f"Attention map extraction failed: {e}")
            
            results[config['name']] = {
                'status': 'success',
                'parameters': model.count_params(),
                'output_shapes': [output.shape for output in test_output] if isinstance(test_output, list) else [test_output.shape]
            }
            
            print("✅ Test passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results[config['name']] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        if result['status'] == 'success':
            print(f"✅ {name}: {result['parameters']:,} parameters")
        else:
            print(f"❌ {name}: {result['error']}")
    
    return results

def test_attention_mechanisms():
    """
    Test individual attention mechanisms
    """
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL ATTENTION MECHANISMS")
    print("="*60)
    
    # Create a simple model to test attention mechanisms
    attention_unet = AttentionUNet(
        input_shape=(64, 64, 1),
        num_classes=1,
        filters=8
    )
    
    # Test input
    test_input = np.random.random((1, 64, 64, 1)).astype(np.float32)
    
    # Test channel attention
    print("Testing Channel Attention...")
    try:
        # Create a simple feature map
        feature_map = tf.constant(test_input)
        attended = attention_unet.channel_attention(feature_map, 8)
        print(f"✅ Channel attention: {feature_map.shape} -> {attended.shape}")
    except Exception as e:
        print(f"❌ Channel attention failed: {e}")
    
    # Test spatial attention
    print("Testing Spatial Attention...")
    try:
        feature_map = tf.constant(test_input)
        attended = attention_unet.spatial_attention(feature_map)
        print(f"✅ Spatial attention: {feature_map.shape} -> {attended.shape}")
    except Exception as e:
        print(f"❌ Spatial attention failed: {e}")
    
    # Test self-attention
    print("Testing Self-Attention...")
    try:
        feature_map = tf.constant(test_input)
        attended = attention_unet.self_attention(feature_map, 8)
        print(f"✅ Self-attention: {feature_map.shape} -> {attended.shape}")
    except Exception as e:
        print(f"❌ Self-attention failed: {e}")

def test_memory_usage():
    """
    Test memory usage of different configurations
    """
    print("\n" + "="*60)
    print("TESTING MEMORY USAGE")
    print("="*60)
    
    import psutil
    import gc
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    configs = [
        {'name': 'Small_Filters', 'filters': 16},
        {'name': 'Medium_Filters', 'filters': 32},
        {'name': 'Large_Filters', 'filters': 64}
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        initial_memory = get_memory_usage()
        
        try:
            attention_unet = AttentionUNet(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                num_classes=1,
                filters=config['filters'],
                use_self_attention=True,
                use_channel_attention=True,
                use_deep_supervision=True
            )
            
            model = attention_unet.build_model()
            model.compile(optimizer=Adam(), loss=BinaryCrossentropy())
            
            final_memory = get_memory_usage()
            memory_used = final_memory - initial_memory
            
            print(f"✅ Memory used: {memory_used:.2f} MB")
            print(f"   Parameters: {model.count_params():,}")
            
            # Clean up
            del model
            del attention_unet
            gc.collect()
            
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    # Run all tests
    test_attention_unet_models()
    test_attention_mechanisms()
    test_memory_usage()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
