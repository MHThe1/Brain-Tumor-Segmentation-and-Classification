"""
Training script for Brain Tumor Segmentation and Classification
BRACU CSE428 Academic Project
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, MeanIoU
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
from models import UNet, AttentionUNet, MultiTaskUNet, EfficientDetUNet, ClassifierModels, AttentionLoss

class BrainTumorTrainer:
    """
    Training class for brain tumor models
    """
    
    def __init__(self, output_dir=RUN_OUTPUT_DIR):
        self.output_dir = output_dir
        self.logger = setup_logging(
            os.path.join(output_dir, "logs", "training.log"),
            LOG_LEVEL
        )
        self.data_loader = BrainTumorDataLoader(IMG_HEIGHT, IMG_WIDTH)
        
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    def train_unet_segmentation(self, model_name="unet_segmentation"):
        """
        Train U-Net for segmentation
        """
        print_experiment_header("U-Net Segmentation Training", 
                              "Training U-Net architecture for brain tumor segmentation")
        
        # Load segmentation data
        (X_train, y_train), (X_val, y_val) = self.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        )
        
        # Create data generators
        train_gen, val_gen = self.data_loader.create_segmentation_generators(
            X_train, y_train, X_val, y_val, BATCH_SIZE
        )
        
        # Build model
        unet = UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                   num_classes=1, filters=32)  # Binary segmentation (memory optimized)
        model = unet.build_model()
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=BinaryCrossentropy(),
            metrics=[dice_coefficient, iou_coefficient]
        )
        
        # Save model summary
        save_model_summary(model, self.output_dir, model_name)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.output_dir, "models", f"{model_name}_best.keras"),
                monitor='val_dice_coefficient',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_dice_coefficient',
                patience=10,
                mode='max',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            CSVLogger(
                os.path.join(self.output_dir, "logs", f"{model_name}_training.csv")
            )
        ]
        
        # Training configuration
        config = {
            'model': 'U-Net',
            'task': 'segmentation',
            'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            'num_classes': SEGMENTATION_CLASSES,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss': 'BinaryCrossentropy',
            'metrics': ['dice_coefficient', 'iou_coefficient']
        }
        
        save_experiment_config(config, self.output_dir)
        
        # Train model
        self.logger.info("Starting U-Net segmentation training...")
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
        
        # Evaluate on test data
        test_images, test_masks = load_test_data(
            self.data_loader, SEGMENTATION_TEST_IMAGES, SEGMENTATION_TEST_MASKS
        )
        
        test_loss, test_dice, test_iou = model.evaluate(test_images, test_masks, verbose=0)
        
        results = {
            'test_loss': float(test_loss),
            'test_dice_coefficient': float(test_dice),
            'test_iou_coefficient': float(test_iou),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'dice_coefficient': [float(x) for x in history.history['dice_coefficient']],
                'val_dice_coefficient': [float(x) for x in history.history['val_dice_coefficient']],
                'iou_coefficient': [float(x) for x in history.history['iou_coefficient']],
                'val_iou_coefficient': [float(x) for x in history.history['val_iou_coefficient']]
            }
        }
        
        log_experiment_results(results, self.output_dir, model_name)
        create_experiment_report(model_name, config, results, self.output_dir)
        
        self.logger.info(f"U-Net segmentation training completed. Test Dice: {test_dice:.4f}, Test IoU: {test_iou:.4f}")
        return model, history, results
    
    def train_attention_unet_segmentation(self, model_name="attention_unet_segmentation"):
        """
        Train Attention U-Net for segmentation
        """
        print_experiment_header("Attention U-Net Segmentation Training", 
                              "Training Attention U-Net architecture for brain tumor segmentation")
        
        # Load segmentation data
        (X_train, y_train), (X_val, y_val) = self.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        )
        
        # Create data generators
        train_gen, val_gen = self.data_loader.create_segmentation_generators(
            X_train, y_train, X_val, y_val, BATCH_SIZE
        )
        
        # Build enhanced model with configurable options
        attention_unet = AttentionUNet(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
            num_classes=1, 
            filters=32,  # Binary segmentation (memory optimized)
            use_self_attention=True,
            use_channel_attention=True,
            use_deep_supervision=True,
            attention_type='both'  # Use both spatial and channel attention
        )
        model = attention_unet.build_model()
        
        # Prepare loss function for deep supervision
        if attention_unet.use_deep_supervision:
            # Deep supervision loss
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
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.output_dir, "models", f"{model_name}_best.keras"),
                monitor='val_dice_coefficient',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_dice_coefficient',
                patience=10,
                mode='max',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            CSVLogger(
                os.path.join(self.output_dir, "logs", f"{model_name}_training.csv")
            )
        ]
        
        # Training configuration
        config = {
            'model': 'Attention U-Net',
            'task': 'segmentation',
            'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            'num_classes': SEGMENTATION_CLASSES,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss': 'BinaryCrossentropy',
            'metrics': ['dice_coefficient', 'iou_coefficient']
        }
        
        save_experiment_config(config, self.output_dir)
        
        # Train model
        self.logger.info("Starting Attention U-Net segmentation training...")
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
        
        # Evaluate on test data
        test_images, test_masks = load_test_data(
            self.data_loader, SEGMENTATION_TEST_IMAGES, SEGMENTATION_TEST_MASKS
        )
        
        test_loss, test_dice, test_iou = model.evaluate(test_images, test_masks, verbose=0)
        
        results = {
            'test_loss': float(test_loss),
            'test_dice_coefficient': float(test_dice),
            'test_iou_coefficient': float(test_iou),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'dice_coefficient': [float(x) for x in history.history['dice_coefficient']],
                'val_dice_coefficient': [float(x) for x in history.history['val_dice_coefficient']],
                'iou_coefficient': [float(x) for x in history.history['iou_coefficient']],
                'val_iou_coefficient': [float(x) for x in history.history['val_iou_coefficient']]
            }
        }
        
        log_experiment_results(results, self.output_dir, model_name)
        create_experiment_report(model_name, config, results, self.output_dir)
        
        self.logger.info(f"Attention U-Net segmentation training completed. Test Dice: {test_dice:.4f}, Test IoU: {test_iou:.4f}")
        return model, history, results
    
    def train_classification(self, classifier_type="simple_cnn", model_name=None):
        """
        Train classifier for brain tumor classification
        """
        if model_name is None:
            model_name = f"{classifier_type}_classification"
        
        print_experiment_header(f"{classifier_type.upper()} Classification Training", 
                              f"Training {classifier_type} for brain tumor classification")
        
        # Load classification data
        (X_train, y_train), (X_val, y_val) = self.data_loader.load_classification_data(
            CLASSIFICATION_TRAIN
        )
        
        # Create data generators
        train_gen, val_gen = self.data_loader.create_data_generators(
            X_train, y_train, X_val, y_val, BATCH_SIZE
        )
        
        # Build model
        if classifier_type == "simple_cnn":
            model = ClassifierModels.simple_cnn(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                num_classes=NUM_CLASSES
            )
        elif classifier_type == "resnet50":
            model = ClassifierModels.resnet50_classifier(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                num_classes=NUM_CLASSES
            )
        elif classifier_type == "efficientnet":
            model = ClassifierModels.efficientnet_classifier(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                num_classes=NUM_CLASSES
            )
        elif classifier_type == "densenet":
            model = ClassifierModels.densenet_classifier(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                num_classes=NUM_CLASSES
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Save model summary
        save_model_summary(model, self.output_dir, model_name)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.output_dir, "models", f"{model_name}_best.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                mode='max',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            CSVLogger(
                os.path.join(self.output_dir, "logs", f"{model_name}_training.csv")
            )
        ]
        
        # Training configuration
        config = {
            'model': classifier_type,
            'task': 'classification',
            'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            'num_classes': NUM_CLASSES,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss': 'CategoricalCrossentropy',
            'metrics': ['accuracy']
        }
        
        save_experiment_config(config, self.output_dir)
        
        # Train model
        self.logger.info(f"Starting {classifier_type} classification training...")
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
        
        # Evaluate on test data (optional - skip if no proper test labels available)
        try:
            test_images, _ = load_test_data(self.data_loader, CLASSIFICATION_TEST)
            
            if len(test_images) > 0:
                # Get test predictions
                test_predictions = model.predict(test_images, verbose=0)
                test_pred_classes = np.argmax(test_predictions, axis=1)
                
                # Note: For proper evaluation, you would need actual test labels
                # For now, we'll just save the predictions without calculating accuracy
                self.logger.info(f"Generated predictions for {len(test_images)} test images")
                
                # Save predictions for manual evaluation
                np.save(os.path.join(self.output_dir, "results", f"{model_name}_test_predictions.npy"), 
                       test_predictions)
                np.save(os.path.join(self.output_dir, "results", f"{model_name}_test_pred_classes.npy"), 
                       test_pred_classes)
                
                # Set dummy test metrics (since we don't have actual labels)
                test_loss = 0.0
                test_accuracy = 0.0
            else:
                self.logger.warning("No test images loaded - skipping test evaluation")
                test_loss = 0.0
                test_accuracy = 0.0
                
        except Exception as e:
            self.logger.error(f"Test evaluation failed: {e}")
            test_loss = 0.0
            test_accuracy = 0.0
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
        }
        
        log_experiment_results(results, self.output_dir, model_name)
        create_experiment_report(model_name, config, results, self.output_dir)
        
        self.logger.info(f"{classifier_type} classification training completed. Test Accuracy: {test_accuracy:.4f}")
        return model, history, results
    
    def train_multitask_model(self, model_name="multitask_unet"):
        """
        Train multi-task model (segmentation + classification)
        """
        print_experiment_header("Multi-Task U-Net Training", 
                              "Training U-Net with both segmentation and classification heads")
        
        # Load both datasets
        (X_seg_train, y_seg_train), (X_seg_val, y_seg_val) = self.data_loader.load_segmentation_data(
            SEGMENTATION_TRAIN_IMAGES, SEGMENTATION_TRAIN_MASKS
        )
        (X_cls_train, y_cls_train), (X_cls_val, y_cls_val) = self.data_loader.load_classification_data(
            CLASSIFICATION_TRAIN
        )
        
        # Build model
        multitask_unet = MultiTaskUNet(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            seg_classes=SEGMENTATION_CLASSES,
            cls_classes=NUM_CLASSES
        )
        model = multitask_unet.build_model()
        
        # Compile model with multiple losses
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss={
                'segmentation': BinaryCrossentropy(),
                'classification': CategoricalCrossentropy()
            },
            loss_weights={'segmentation': 1.0, 'classification': 1.0},
            metrics={
                'segmentation': [dice_coefficient, iou_coefficient],
                'classification': ['accuracy']
            }
        )
        
        # Save model summary
        save_model_summary(model, self.output_dir, model_name)
        
        # For multi-task training, we need to create a custom data generator
        # This is a simplified version - in practice you'd need more sophisticated data handling
        def multitask_generator(X_seg, y_seg, X_cls, y_cls, batch_size):
            while True:
                indices = np.random.choice(len(X_seg), batch_size, replace=False)
                batch_X = X_seg[indices]
                batch_y_seg = y_seg[indices]
                
                # For classification, we'll use the same images (simplified)
                batch_y_cls = y_cls[indices % len(y_cls)]
                
                yield batch_X, {'segmentation': batch_y_seg, 'classification': batch_y_cls}
        
        train_gen = multitask_generator(X_seg_train, y_seg_train, X_cls_train, y_cls_train, BATCH_SIZE)
        val_gen = multitask_generator(X_seg_val, y_seg_val, X_cls_val, y_cls_val, BATCH_SIZE)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.output_dir, "models", f"{model_name}_best.keras"),
                monitor='val_segmentation_dice_coefficient',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_segmentation_dice_coefficient',
                patience=15,
                mode='max',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            CSVLogger(
                os.path.join(self.output_dir, "logs", f"{model_name}_training.csv")
            )
        ]
        
        # Training configuration
        config = {
            'model': 'Multi-Task U-Net',
            'task': 'segmentation + classification',
            'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            'seg_classes': SEGMENTATION_CLASSES,
            'cls_classes': NUM_CLASSES,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss_weights': {'segmentation': 1.0, 'classification': 1.0}
        }
        
        save_experiment_config(config, self.output_dir)
        
        # Train model
        self.logger.info("Starting multi-task U-Net training...")
        history = model.fit(
            train_gen,
            steps_per_epoch=len(X_seg_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=len(X_seg_val) // BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(os.path.join(self.output_dir, "models", f"{model_name}_final.keras"))
        
        # Plot training history
        plot_training_history(history, self.output_dir, model_name)
        
        # Evaluate on test data
        test_images_seg, test_masks = load_test_data(
            self.data_loader, SEGMENTATION_TEST_IMAGES, SEGMENTATION_TEST_MASKS
        )
        
        test_results = model.evaluate(
            test_images_seg, 
            {'segmentation': test_masks, 'classification': np.zeros((len(test_images_seg), NUM_CLASSES))},
            verbose=0
        )
        
        results = {
            'test_results': test_results,
            'training_history': history.history
        }
        
        log_experiment_results(results, self.output_dir, model_name)
        create_experiment_report(model_name, config, results, self.output_dir)
        
        self.logger.info("Multi-task U-Net training completed.")
        return model, history, results

def main():
    """
    Main training function
    """
    print("="*80)
    print("BRAIN TUMOR SEGMENTATION AND CLASSIFICATION TRAINING")
    print("BRACU CSE428 Academic Project")
    print("="*80)
    
    # Initialize trainer
    trainer = BrainTumorTrainer()
    
    # Training experiments
    experiments = [
        ("U-Net Segmentation", trainer.train_unet_segmentation),
        ("Attention U-Net Segmentation", trainer.train_attention_unet_segmentation),
        ("Simple CNN Classification", lambda: trainer.train_classification("simple_cnn")),
        ("ResNet50 Classification", lambda: trainer.train_classification("resnet50")),
        ("Multi-Task U-Net", trainer.train_multitask_model)
    ]
    
    results_summary = {}
    
    for exp_name, exp_func in experiments:
        try:
            print(f"\n{'='*60}")
            print(f"Starting experiment: {exp_name}")
            print(f"{'='*60}")
            
            model, history, results = exp_func()
            results_summary[exp_name] = {
                'status': 'completed',
                'results': results
            }
            
        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")
            results_summary[exp_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save overall results summary
    with open(os.path.join(RUN_OUTPUT_DIR, "results", "experiments_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=4, default=str)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {RUN_OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
