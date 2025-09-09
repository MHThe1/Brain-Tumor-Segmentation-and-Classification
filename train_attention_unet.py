"""
Enhanced Training script for Attention U-Net Brain Tumor Segmentation
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
    except RuntimeError as e:
        print(f"Memory growth setup failed: {e}")

# Import project modules
from config import *
from utils import *
from data_loader import BrainTumorDataLoader, load_test_data
from models import AttentionUNet, AttentionLoss

class EnhancedAttentionUNetTrainer:
    """
    Enhanced training class for Attention U-Net with advanced features
    """
    
    def __init__(self, output_dir=RUN_OUTPUT_DIR):
        self.output_dir = output_dir
        self.logger = setup_logging(
            os.path.join(output_dir, "logs", "attention_unet_training.log"),
            LOG_LEVEL
        )
        self.data_loader = BrainTumorDataLoader(IMG_HEIGHT, IMG_WIDTH)
        
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "attention_maps"), exist_ok=True)
    
    def train_attention_unet(self, model_name="enhanced_attention_unet", 
                           use_self_attention=True, use_channel_attention=True,
                           use_deep_supervision=True, attention_type='both'):
        """
        Train Enhanced Attention U-Net with configurable options
        """
        print_experiment_header("Enhanced Attention U-Net Training", 
                              f"Training with self_attention={use_self_attention}, "
                              f"channel_attention={use_channel_attention}, "
                              f"deep_supervision={use_deep_supervision}, "
                              f"attention_type={attention_type}")
        
        # Load segmentation data
        (X_train, y_train), (X_val, y_val) = self.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        )
        
        # Create data generators
        train_gen, val_gen = self.data_loader.create_segmentation_generators(
            X_train, y_train, X_val, y_val, BATCH_SIZE
        )
        
        # Build enhanced model with memory optimization
        # Reduce filters for self-attention models to prevent OOM
        if use_self_attention:
            filter_size = 16  # Smaller for self-attention
        else:
            filter_size = 32  # Standard size
            
        attention_unet = AttentionUNet(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
            num_classes=1, 
            filters=filter_size,  # Memory optimized
            use_self_attention=use_self_attention,
            use_channel_attention=use_channel_attention,
            use_deep_supervision=use_deep_supervision,
            attention_type=attention_type
        )
        model = attention_unet.build_model()
        
        # Prepare loss function and metrics
        if use_deep_supervision:
            loss_fn = AttentionLoss.deep_supervision_loss(alpha=0.4, beta=0.3, gamma=0.3)
            metrics = {
                'main_output': [dice_coefficient, iou_coefficient],
                'aux_output_4': [dice_coefficient, iou_coefficient],
                'aux_output_3': [dice_coefficient, iou_coefficient],
                'aux_output_2': [dice_coefficient, iou_coefficient]
            }
        else:
            loss_fn = BinaryCrossentropy()
            metrics = [dice_coefficient, iou_coefficient]
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=loss_fn,
            metrics=metrics
        )
        
        # Save model summary
        save_model_summary(model, self.output_dir, model_name)
        
        # Enhanced callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.output_dir, "models", f"{model_name}_best.keras"),
                monitor='val_main_output_dice_coefficient' if use_deep_supervision else 'val_dice_coefficient',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_main_output_dice_coefficient' if use_deep_supervision else 'val_dice_coefficient',
                patience=15,  # Increased patience for complex model
                mode='max',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                os.path.join(self.output_dir, "logs", f"{model_name}_training.csv")
            )
        ]
        
        # Training configuration
        config = {
            'model': 'Enhanced Attention U-Net',
            'task': 'segmentation',
            'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            'num_classes': 1,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss': 'Deep Supervision Loss' if use_deep_supervision else 'BinaryCrossentropy',
            'use_self_attention': use_self_attention,
            'use_channel_attention': use_channel_attention,
            'use_deep_supervision': use_deep_supervision,
            'attention_type': attention_type,
            'metrics': ['dice_coefficient', 'iou_coefficient']
        }
        
        save_experiment_config(config, self.output_dir)
        
        # Train model
        self.logger.info("Starting Enhanced Attention U-Net training...")
        history = model.fit(
            train_gen,
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=len(X_val) // BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(os.path.join(self.output_dir, "models", f"{model_name}_final.keras"))
        
        # Plot training history
        plot_training_history(history, self.output_dir, model_name)
        
        # Evaluate on test data with memory optimization
        test_images, test_masks = load_test_data(
            self.data_loader, SEGMENTATION_TEST_IMAGES, SEGMENTATION_TEST_MASKS
        )
        
        # Limit test data size to prevent OOM
        max_test_samples = 100  # Reduce from 860 to prevent memory issues
        if len(test_images) > max_test_samples:
            test_images = test_images[:max_test_samples]
            test_masks = test_masks[:max_test_samples]
            self.logger.info(f"Limited test data to {max_test_samples} samples to prevent OOM")
        
        if use_deep_supervision:
            test_results = model.evaluate(test_images, test_masks, verbose=0)
            test_loss = test_results[0]
            test_dice = test_results[1]  # Main output dice
            test_iou = test_results[2]   # Main output iou
        else:
            test_loss, test_dice, test_iou = model.evaluate(test_images, test_masks, verbose=0)
        
        # Generate attention visualizations
        self.generate_attention_visualizations(model, attention_unet, test_images, model_name)
        
        # Generate comprehensive results
        results = self.generate_results(history, test_loss, test_dice, test_iou, use_deep_supervision)
        
        log_experiment_results(results, self.output_dir, model_name)
        create_experiment_report(model_name, config, results, self.output_dir)
        
        self.logger.info(f"Enhanced Attention U-Net training completed. Test Dice: {test_dice:.4f}, Test IoU: {test_iou:.4f}")
        return model, history, results
    
    def generate_attention_visualizations(self, model, attention_unet, test_images, model_name):
        """
        Generate comprehensive attention visualizations
        """
        try:
            self.logger.info("Generating attention visualizations...")
            
            # Visualize attention maps for multiple test images
            num_samples = min(5, len(test_images))
            for i in range(num_samples):
                sample_image = test_images[i:i+1]
                
                # Generate attention maps
                attention_maps = attention_unet.get_attention_maps(model, sample_image)
                
                # Create visualization
                fig, axes = plt.subplots(2, len(attention_maps) + 1, figsize=(20, 8))
                if len(attention_maps) == 0:
                    axes = axes.reshape(2, 1)
                
                # Original image
                axes[0, 0].imshow(sample_image[0, :, :, 0], cmap='gray')
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                
                axes[1, 0].imshow(sample_image[0, :, :, 0], cmap='gray')
                axes[1, 0].set_title('Original Image')
                axes[1, 0].axis('off')
                
                # Attention maps
                for j, (layer_name, attention_map) in enumerate(attention_maps.items()):
                    if len(attention_map.shape) == 4:
                        attention_map = attention_map[0, :, :, 0]
                    
                    # Overlay attention on original image
                    axes[0, j+1].imshow(sample_image[0, :, :, 0], cmap='gray', alpha=0.7)
                    axes[0, j+1].imshow(attention_map, cmap='hot', alpha=0.5)
                    axes[0, j+1].set_title(f'Attention Overlay - {layer_name}')
                    axes[0, j+1].axis('off')
                    
                    # Pure attention map
                    axes[1, j+1].imshow(attention_map, cmap='hot')
                    axes[1, j+1].set_title(f'Attention Map - {layer_name}')
                    axes[1, j+1].axis('off')
                
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_dir, "attention_maps", f"{model_name}_attention_sample_{i}.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
            
            self.logger.info(f"Generated attention visualizations for {num_samples} samples")
            
        except Exception as e:
            self.logger.error(f"Attention visualization failed: {e}")
    
    def generate_results(self, history, test_loss, test_dice, test_iou, use_deep_supervision):
        """
        Generate comprehensive results dictionary
        """
        results = {
            'test_loss': float(test_loss),
            'test_dice_coefficient': float(test_dice),
            'test_iou_coefficient': float(test_iou),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
            }
        }
        
        # Add metrics based on deep supervision
        if use_deep_supervision:
            results['training_history'].update({
                'main_output_dice_coefficient': [float(x) for x in history.history['main_output_dice_coefficient']],
                'val_main_output_dice_coefficient': [float(x) for x in history.history['val_main_output_dice_coefficient']],
                'main_output_iou_coefficient': [float(x) for x in history.history['main_output_iou_coefficient']],
                'val_main_output_iou_coefficient': [float(x) for x in history.history['val_main_output_iou_coefficient']],
            })
        else:
            results['training_history'].update({
                'dice_coefficient': [float(x) for x in history.history['dice_coefficient']],
                'val_dice_coefficient': [float(x) for x in history.history['val_dice_coefficient']],
                'iou_coefficient': [float(x) for x in history.history['iou_coefficient']],
                'val_iou_coefficient': [float(x) for x in history.history['val_iou_coefficient']]
            })
        
        return results

def main():
    """
    Main training function for Enhanced Attention U-Net
    """
    print("="*80)
    print("ENHANCED ATTENTION U-NET BRAIN TUMOR SEGMENTATION TRAINING")
    print("BRACU CSE428 Academic Project")
    print("="*80)
    
    # Initialize trainer
    trainer = EnhancedAttentionUNetTrainer()
    
    # Training experiments with different configurations
    experiments = [
        {
            'name': 'Enhanced_Attention_UNet_Full',
            'use_self_attention': True,
            'use_channel_attention': True,
            'use_deep_supervision': True,
            'attention_type': 'both'
        },
        {
            'name': 'Enhanced_Attention_UNet_Spatial_Only',
            'use_self_attention': False,
            'use_channel_attention': False,
            'use_deep_supervision': False,
            'attention_type': 'spatial'
        },
        {
            'name': 'Enhanced_Attention_UNet_Channel_Only',
            'use_self_attention': False,
            'use_channel_attention': True,
            'use_deep_supervision': False,
            'attention_type': 'channel'
        }
    ]
    
    results_summary = {}
    
    for exp_config in experiments:
        try:
            print(f"\n{'='*60}")
            print(f"Starting experiment: {exp_config['name']}")
            print(f"{'='*60}")
            
            model, history, results = trainer.train_attention_unet(
                model_name=exp_config['name'],
                use_self_attention=exp_config['use_self_attention'],
                use_channel_attention=exp_config['use_channel_attention'],
                use_deep_supervision=exp_config['use_deep_supervision'],
                attention_type=exp_config['attention_type']
            )
            
            results_summary[exp_config['name']] = {
                'status': 'completed',
                'results': results,
                'config': exp_config
            }
            
        except Exception as e:
            print(f"Error in experiment {exp_config['name']}: {e}")
            results_summary[exp_config['name']] = {
                'status': 'failed',
                'error': str(e),
                'config': exp_config
            }
    
    # Save overall results summary
    with open(os.path.join(RUN_OUTPUT_DIR, "results", "enhanced_attention_unet_experiments_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=4, default=str)
    
    print(f"\n{'='*80}")
    print("ALL ENHANCED ATTENTION U-NET EXPERIMENTS COMPLETED")
    print(f"Results saved to: {RUN_OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
