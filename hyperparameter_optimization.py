"""
Hyperparameter Optimization for Brain Tumor Models
BRACU CSE428 Academic Project - Bonus Task

This script implements hyperparameter optimization for:
1. Learning rate optimization
2. Optimizer comparison
3. Architecture parameter tuning
4. Loss function weight optimization
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import itertools
from datetime import datetime
import logging

# Import project modules
from config import *
from utils import *
from data_loader import BrainTumorDataLoader
from models import UNet, AttentionUNet, MultiTaskUNet

class HyperparameterOptimizer:
    """
    Hyperparameter optimization class for brain tumor models
    """
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(RUN_OUTPUT_DIR, "hyperparameter_optimization")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = setup_logging(
            os.path.join(self.output_dir, "optimization.log"),
            LOG_LEVEL
        )
        
        self.data_loader = BrainTumorDataLoader(IMG_HEIGHT, IMG_WIDTH)
        self.results = []
    
    def optimize_learning_rate(self, model_type="unet", epochs=20):
        """
        Optimize learning rate for the model
        """
        print_experiment_header("Learning Rate Optimization", 
                              f"Finding optimal learning rate for {model_type}")
        
        # Learning rates to test
        learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        
        # Load data
        (X_train, y_train), (X_val, y_val) = self.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        )
        
        train_gen, val_gen = self.data_loader.create_segmentation_generators(
            X_train, y_train, X_val, y_val, BATCH_SIZE
        )
        
        lr_results = []
        
        for lr in learning_rates:
            self.logger.info(f"Testing learning rate: {lr}")
            
            # Build model
            if model_type == "unet":
                model_builder = UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                   num_classes=SEGMENTATION_CLASSES)
            elif model_type == "attention_unet":
                model_builder = AttentionUNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                            num_classes=SEGMENTATION_CLASSES)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = model_builder.build_model()
            
            # Compile with current learning rate
            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss=BinaryCrossentropy(),
                metrics=[dice_coefficient, iou_coefficient]
            )
            
            # Train for limited epochs
            history = model.fit(
                train_gen,
                steps_per_epoch=len(X_train) // BATCH_SIZE,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=len(X_val) // BATCH_SIZE,
                verbose=0
            )
            
            # Get best validation dice coefficient
            best_val_dice = max(history.history['val_dice_coefficient'])
            best_val_iou = max(history.history['val_iou_coefficient'])
            
            result = {
                'learning_rate': lr,
                'best_val_dice': float(best_val_dice),
                'best_val_iou': float(best_val_iou),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
            
            lr_results.append(result)
            self.logger.info(f"LR {lr}: Val Dice = {best_val_dice:.4f}, Val IoU = {best_val_iou:.4f}")
        
        # Find best learning rate
        best_lr_result = max(lr_results, key=lambda x: x['best_val_dice'])
        best_lr = best_lr_result['learning_rate']
        
        self.logger.info(f"Best learning rate: {best_lr} with Dice: {best_lr_result['best_val_dice']:.4f}")
        
        # Save results
        lr_results_data = {
            'experiment': 'learning_rate_optimization',
            'model_type': model_type,
            'learning_rates_tested': learning_rates,
            'results': lr_results,
            'best_learning_rate': best_lr,
            'best_result': best_lr_result
        }
        
        with open(os.path.join(self.output_dir, f"lr_optimization_{model_type}.json"), 'w') as f:
            json.dump(lr_results_data, f, indent=4)
        
        return best_lr, lr_results
    
    def optimize_optimizer(self, model_type="unet", learning_rate=1e-3, epochs=20):
        """
        Compare different optimizers
        """
        print_experiment_header("Optimizer Comparison", 
                              f"Comparing optimizers for {model_type}")
        
        optimizers = {
            'Adam': Adam(learning_rate=learning_rate),
            'SGD': SGD(learning_rate=learning_rate, momentum=0.9),
            'RMSprop': RMSprop(learning_rate=learning_rate)
        }
        
        # Load data
        (X_train, y_train), (X_val, y_val) = self.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        )
        
        train_gen, val_gen = self.data_loader.create_segmentation_generators(
            X_train, y_train, X_val, y_val, BATCH_SIZE
        )
        
        optimizer_results = []
        
        for opt_name, optimizer in optimizers.items():
            self.logger.info(f"Testing optimizer: {opt_name}")
            
            # Build model
            if model_type == "unet":
                model_builder = UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                   num_classes=SEGMENTATION_CLASSES)
            elif model_type == "attention_unet":
                model_builder = AttentionUNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                            num_classes=SEGMENTATION_CLASSES)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = model_builder.build_model()
            
            # Compile with current optimizer
            model.compile(
                optimizer=optimizer,
                loss=BinaryCrossentropy(),
                metrics=[dice_coefficient, iou_coefficient]
            )
            
            # Train for limited epochs
            history = model.fit(
                train_gen,
                steps_per_epoch=len(X_train) // BATCH_SIZE,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=len(X_val) // BATCH_SIZE,
                verbose=0
            )
            
            # Get best validation metrics
            best_val_dice = max(history.history['val_dice_coefficient'])
            best_val_iou = max(history.history['val_iou_coefficient'])
            final_val_loss = history.history['val_loss'][-1]
            
            result = {
                'optimizer': opt_name,
                'best_val_dice': float(best_val_dice),
                'best_val_iou': float(best_val_iou),
                'final_val_loss': float(final_val_loss),
                'convergence_epoch': np.argmax(history.history['val_dice_coefficient']) + 1
            }
            
            optimizer_results.append(result)
            self.logger.info(f"{opt_name}: Val Dice = {best_val_dice:.4f}, Val IoU = {best_val_iou:.4f}")
        
        # Find best optimizer
        best_opt_result = max(optimizer_results, key=lambda x: x['best_val_dice'])
        best_optimizer = best_opt_result['optimizer']
        
        self.logger.info(f"Best optimizer: {best_optimizer} with Dice: {best_opt_result['best_val_dice']:.4f}")
        
        # Save results
        opt_results_data = {
            'experiment': 'optimizer_comparison',
            'model_type': model_type,
            'learning_rate': learning_rate,
            'optimizers_tested': list(optimizers.keys()),
            'results': optimizer_results,
            'best_optimizer': best_optimizer,
            'best_result': best_opt_result
        }
        
        with open(os.path.join(self.output_dir, f"optimizer_comparison_{model_type}.json"), 'w') as f:
            json.dump(opt_results_data, f, indent=4)
        
        return best_optimizer, optimizer_results
    
    def optimize_architecture_parameters(self, model_type="unet", epochs=15):
        """
        Optimize architecture parameters (filters, depth)
        """
        print_experiment_header("Architecture Parameter Optimization", 
                              f"Optimizing architecture for {model_type}")
        
        # Parameter combinations to test
        filter_sizes = [32, 64, 96]
        learning_rate = 1e-3  # Use a reasonable default
        
        # Load data
        (X_train, y_train), (X_val, y_val) = self.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        )
        
        train_gen, val_gen = self.data_loader.create_segmentation_generators(
            X_train, y_train, X_val, y_val, BATCH_SIZE
        )
        
        arch_results = []
        
        for filters in filter_sizes:
            self.logger.info(f"Testing filter size: {filters}")
            
            # Build model with current parameters
            if model_type == "unet":
                model_builder = UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                   num_classes=SEGMENTATION_CLASSES, filters=filters)
            elif model_type == "attention_unet":
                model_builder = AttentionUNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                            num_classes=SEGMENTATION_CLASSES, filters=filters)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = model_builder.build_model()
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=BinaryCrossentropy(),
                metrics=[dice_coefficient, iou_coefficient]
            )
            
            # Train for limited epochs
            history = model.fit(
                train_gen,
                steps_per_epoch=len(X_train) // BATCH_SIZE,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=len(X_val) // BATCH_SIZE,
                verbose=0
            )
            
            # Get best validation metrics
            best_val_dice = max(history.history['val_dice_coefficient'])
            best_val_iou = max(history.history['val_iou_coefficient'])
            param_count = model.count_params()
            
            result = {
                'filters': filters,
                'parameters': param_count,
                'best_val_dice': float(best_val_dice),
                'best_val_iou': float(best_val_iou),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
            
            arch_results.append(result)
            self.logger.info(f"Filters {filters}: Params = {param_count:,}, Val Dice = {best_val_dice:.4f}")
        
        # Find best architecture
        best_arch_result = max(arch_results, key=lambda x: x['best_val_dice'])
        best_filters = best_arch_result['filters']
        
        self.logger.info(f"Best filter size: {best_filters} with Dice: {best_arch_result['best_val_dice']:.4f}")
        
        # Save results
        arch_results_data = {
            'experiment': 'architecture_optimization',
            'model_type': model_type,
            'filter_sizes_tested': filter_sizes,
            'results': arch_results,
            'best_filters': best_filters,
            'best_result': best_arch_result
        }
        
        with open(os.path.join(self.output_dir, f"architecture_optimization_{model_type}.json"), 'w') as f:
            json.dump(arch_results_data, f, indent=4)
        
        return best_filters, arch_results
    
    def optimize_multitask_weights(self, epochs=15):
        """
        Optimize loss weights for multi-task learning
        """
        print_experiment_header("Multi-Task Loss Weight Optimization", 
                              "Finding optimal loss weights for segmentation and classification")
        
        # Loss weight combinations to test
        weight_combinations = [
            {'segmentation': 1.0, 'classification': 0.5},
            {'segmentation': 1.0, 'classification': 1.0},
            {'segmentation': 1.0, 'classification': 2.0},
            {'segmentation': 2.0, 'classification': 1.0},
            {'segmentation': 0.5, 'classification': 1.0}
        ]
        
        # Load both datasets
        (X_seg_train, y_seg_train), (X_seg_val, y_seg_val) = self.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        )
        (X_cls_train, y_cls_train), (X_cls_val, y_cls_val) = self.data_loader.load_classification_data(
            CLASSIFICATION_TRAIN
        )
        
        multitask_results = []
        
        for weights in weight_combinations:
            self.logger.info(f"Testing weights: {weights}")
            
            # Build multi-task model
            multitask_unet = MultiTaskUNet(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                seg_classes=SEGMENTATION_CLASSES,
                cls_classes=NUM_CLASSES
            )
            model = multitask_unet.build_model()
            
            # Compile with current weights
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss={
                    'segmentation': BinaryCrossentropy(),
                    'classification': CategoricalCrossentropy()
                },
                loss_weights=weights,
                metrics={
                    'segmentation': [dice_coefficient, iou_coefficient],
                    'classification': ['accuracy']
                }
            )
            
            # Create multi-task data generator (simplified)
            def multitask_generator(X_seg, y_seg, X_cls, y_cls, batch_size):
                while True:
                    indices = np.random.choice(len(X_seg), batch_size, replace=False)
                    batch_X = X_seg[indices]
                    batch_y_seg = y_seg[indices]
                    batch_y_cls = y_cls[indices % len(y_cls)]
                    yield batch_X, {'segmentation': batch_y_seg, 'classification': batch_y_cls}
            
            train_gen = multitask_generator(X_seg_train, y_seg_train, X_cls_train, y_cls_train, BATCH_SIZE)
            val_gen = multitask_generator(X_seg_val, y_seg_val, X_cls_val, y_cls_val, BATCH_SIZE)
            
            # Train for limited epochs
            history = model.fit(
                train_gen,
                steps_per_epoch=len(X_seg_train) // BATCH_SIZE,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=len(X_seg_val) // BATCH_SIZE,
                verbose=0
            )
            
            # Get best validation metrics
            best_val_dice = max(history.history['val_segmentation_dice_coefficient'])
            best_val_cls_acc = max(history.history['val_classification_accuracy'])
            combined_score = best_val_dice * 0.7 + best_val_cls_acc * 0.3  # Weighted combination
            
            result = {
                'segmentation_weight': weights['segmentation'],
                'classification_weight': weights['classification'],
                'best_val_dice': float(best_val_dice),
                'best_val_cls_accuracy': float(best_val_cls_acc),
                'combined_score': float(combined_score)
            }
            
            multitask_results.append(result)
            self.logger.info(f"Weights {weights}: Dice = {best_val_dice:.4f}, Cls Acc = {best_val_cls_acc:.4f}")
        
        # Find best weight combination
        best_weight_result = max(multitask_results, key=lambda x: x['combined_score'])
        best_weights = {
            'segmentation': best_weight_result['segmentation_weight'],
            'classification': best_weight_result['classification_weight']
        }
        
        self.logger.info(f"Best weights: {best_weights} with combined score: {best_weight_result['combined_score']:.4f}")
        
        # Save results
        weight_results_data = {
            'experiment': 'multitask_weight_optimization',
            'weight_combinations_tested': weight_combinations,
            'results': multitask_results,
            'best_weights': best_weights,
            'best_result': best_weight_result
        }
        
        with open(os.path.join(self.output_dir, "multitask_weight_optimization.json"), 'w') as f:
            json.dump(weight_results_data, f, indent=4)
        
        return best_weights, multitask_results
    
    def run_comprehensive_optimization(self):
        """
        Run comprehensive hyperparameter optimization
        """
        print_experiment_header("Comprehensive Hyperparameter Optimization", 
                              "Running all optimization experiments")
        
        optimization_results = {}
        
        # 1. Learning rate optimization
        try:
            best_lr_unet, lr_results_unet = self.optimize_learning_rate("unet", epochs=15)
            best_lr_att, lr_results_att = self.optimize_learning_rate("attention_unet", epochs=15)
            
            optimization_results['learning_rate'] = {
                'unet': {'best_lr': best_lr_unet, 'results': lr_results_unet},
                'attention_unet': {'best_lr': best_lr_att, 'results': lr_results_att}
            }
        except Exception as e:
            self.logger.error(f"Learning rate optimization failed: {e}")
        
        # 2. Optimizer comparison
        try:
            best_opt_unet, opt_results_unet = self.optimize_optimizer("unet", epochs=15)
            best_opt_att, opt_results_att = self.optimize_optimizer("attention_unet", epochs=15)
            
            optimization_results['optimizer'] = {
                'unet': {'best_optimizer': best_opt_unet, 'results': opt_results_unet},
                'attention_unet': {'best_optimizer': best_opt_att, 'results': opt_results_att}
            }
        except Exception as e:
            self.logger.error(f"Optimizer comparison failed: {e}")
        
        # 3. Architecture optimization
        try:
            best_filters_unet, arch_results_unet = self.optimize_architecture_parameters("unet", epochs=10)
            best_filters_att, arch_results_att = self.optimize_architecture_parameters("attention_unet", epochs=10)
            
            optimization_results['architecture'] = {
                'unet': {'best_filters': best_filters_unet, 'results': arch_results_unet},
                'attention_unet': {'best_filters': best_filters_att, 'results': arch_results_att}
            }
        except Exception as e:
            self.logger.error(f"Architecture optimization failed: {e}")
        
        # 4. Multi-task weight optimization
        try:
            best_weights, weight_results = self.optimize_multitask_weights(epochs=10)
            optimization_results['multitask_weights'] = {
                'best_weights': best_weights,
                'results': weight_results
            }
        except Exception as e:
            self.logger.error(f"Multi-task weight optimization failed: {e}")
        
        # Save comprehensive results
        comprehensive_results = {
            'experiment_info': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'description': 'Comprehensive hyperparameter optimization',
                'models_tested': ['unet', 'attention_unet', 'multitask_unet']
            },
            'optimization_results': optimization_results
        }
        
        with open(os.path.join(self.output_dir, "comprehensive_optimization_results.json"), 'w') as f:
            json.dump(comprehensive_results, f, indent=4, default=str)
        
        # Create summary report
        self.create_optimization_report(optimization_results)
        
        return optimization_results
    
    def create_optimization_report(self, results):
        """
        Create a comprehensive optimization report
        """
        report_path = os.path.join(self.output_dir, "optimization_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Hyperparameter Optimization Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report summarizes the results of comprehensive hyperparameter optimization ")
            f.write("for brain tumor segmentation and classification models.\n\n")
            
            # Learning rate results
            if 'learning_rate' in results:
                f.write("## Learning Rate Optimization\n\n")
                for model_type, data in results['learning_rate'].items():
                    f.write(f"### {model_type.upper()}\n")
                    f.write(f"- **Best Learning Rate:** {data['best_lr']}\n")
                    f.write(f"- **Best Validation Dice:** {max([r['best_val_dice'] for r in data['results']]):.4f}\n\n")
            
            # Optimizer results
            if 'optimizer' in results:
                f.write("## Optimizer Comparison\n\n")
                for model_type, data in results['optimizer'].items():
                    f.write(f"### {model_type.upper()}\n")
                    f.write(f"- **Best Optimizer:** {data['best_optimizer']}\n")
                    f.write(f"- **Best Validation Dice:** {max([r['best_val_dice'] for r in data['results']]):.4f}\n\n")
            
            # Architecture results
            if 'architecture' in results:
                f.write("## Architecture Optimization\n\n")
                for model_type, data in results['architecture'].items():
                    f.write(f"### {model_type.upper()}\n")
                    f.write(f"- **Best Filter Size:** {data['best_filters']}\n")
                    f.write(f"- **Best Validation Dice:** {max([r['best_val_dice'] for r in data['results']]):.4f}\n\n")
            
            # Multi-task results
            if 'multitask_weights' in results:
                f.write("## Multi-Task Weight Optimization\n\n")
                data = results['multitask_weights']
                f.write(f"- **Best Weights:** Segmentation={data['best_weights']['segmentation']}, ")
                f.write(f"Classification={data['best_weights']['classification']}\n")
                f.write(f"- **Best Combined Score:** {max([r['combined_score'] for r in data['results']]):.4f}\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the optimization results:\n\n")
            f.write("1. **Learning Rate:** Use the optimized learning rates for each model\n")
            f.write("2. **Optimizer:** Use the best performing optimizer for each architecture\n")
            f.write("3. **Architecture:** Use the optimal filter sizes for better performance\n")
            f.write("4. **Multi-Task:** Use the optimized loss weights for joint training\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Train final models with optimized hyperparameters\n")
            f.write("2. Compare performance with baseline models\n")
            f.write("3. Conduct statistical significance tests\n")
            f.write("4. Document final model configurations\n")
        
        self.logger.info(f"Optimization report saved to: {report_path}")

def main():
    """
    Main hyperparameter optimization function
    """
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("BRACU CSE428 Academic Project - Bonus Task")
    print("="*80)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer()
    
    # Run comprehensive optimization
    results = optimizer.run_comprehensive_optimization()
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print(f"Results saved to: {optimizer.output_dir}")
    print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    main()
