"""
Data loading and preprocessing utilities for Brain Tumor Segmentation and Classification
BRACU CSE428 Academic Project
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import logging
from config import *

class BrainTumorDataLoader:
    """
    Data loader class for brain tumor dataset
    """
    
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
        self.img_height = img_height
        self.img_width = img_width
        self.logger = logging.getLogger(__name__)
        
        # Class mappings
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def load_classification_data(self, data_dir, test_size=0.2, random_state=42):
        """
        Load classification dataset
        """
        self.logger.info("Loading classification dataset...")
        
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            self.logger.info(f"Loading {class_name} images...")
            class_images = []
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        # Load and preprocess image
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (self.img_width, self.img_height))
                        image = image.astype(np.float32) / 255.0
                        
                        class_images.append(image)
                        labels.append(self.class_to_idx[class_name])
                        
                    except Exception as e:
                        self.logger.error(f"Error loading image {img_path}: {e}")
                        continue
            
            images.extend(class_images)
            self.logger.info(f"Loaded {len(class_images)} {class_name} images")
        
        images = np.array(images)
        labels = np.array(labels)
        
        self.logger.info(f"Total images loaded: {len(images)}")
        self.logger.info(f"Image shape: {images.shape}")
        self.logger.info(f"Labels shape: {labels.shape}")
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=len(self.class_names))
        y_val_cat = to_categorical(y_val, num_classes=len(self.class_names))
        
        return (X_train, y_train_cat), (X_val, y_val_cat)
    
    def load_segmentation_data(self, images_dir, masks_dir, test_size=0.2, random_state=42):
        """
        Load segmentation dataset
        """
        self.logger.info("Loading segmentation dataset...")
        
        # Get list of image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Match image and mask files
        matched_files = []
        for img_file in image_files:
            # Try different mask file extensions
            mask_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = img_file.rsplit('.', 1)[0] + ext
                if potential_mask in mask_files:
                    mask_file = potential_mask
                    break
            
            if mask_file:
                matched_files.append((img_file, mask_file))
        
        self.logger.info(f"Found {len(matched_files)} matched image-mask pairs")
        
        images = []
        masks = []
        
        for img_file, mask_file in matched_files:
            try:
                # Load image
                img_path = os.path.join(images_dir, img_file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.img_width, self.img_height))
                image = image.astype(np.float32) / 255.0
                
                # Load mask
                mask_path = os.path.join(masks_dir, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.img_width, self.img_height))
                mask = (mask > 128).astype(np.float32)  # Binarize mask
                
                images.append(image)
                masks.append(mask)
                
            except Exception as e:
                self.logger.error(f"Error loading {img_file}/{mask_file}: {e}")
                continue
        
        images = np.array(images)
        masks = np.array(masks)
        
        # Add channel dimension to masks
        masks = np.expand_dims(masks, axis=-1)
        
        self.logger.info(f"Total segmentation pairs loaded: {len(images)}")
        self.logger.info(f"Images shape: {images.shape}")
        self.logger.info(f"Masks shape: {masks.shape}")
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            images, masks, test_size=test_size, random_state=random_state
        )
        
        return (X_train, y_train), (X_val, y_val)
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
        """
        Create data generators with augmentation
        """
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def create_segmentation_generators(self, X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
        """
        Create data generators for segmentation with augmentation
        """
        # Custom data generator for segmentation
        def segmentation_generator(X, y, batch_size, augment=True):
            while True:
                indices = np.random.choice(len(X), batch_size, replace=False)
                batch_X = X[indices]
                batch_y = y[indices]
                
                if augment:
                    # Apply augmentation
                    for i in range(batch_size):
                        # Random rotation
                        angle = np.random.uniform(-15, 15)
                        h, w = batch_X[i].shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        batch_X[i] = cv2.warpAffine(batch_X[i], M, (w, h))
                        
                        # Handle mask shape for rotation
                        if len(batch_y[i].shape) == 3:
                            # Mask has shape (h, w, 1), squeeze to (h, w) for cv2
                            mask_2d = np.squeeze(batch_y[i], axis=-1)
                            mask_rotated = cv2.warpAffine(mask_2d, M, (w, h))
                            batch_y[i] = np.expand_dims(mask_rotated, axis=-1)
                        else:
                            batch_y[i] = cv2.warpAffine(batch_y[i], M, (w, h))
                        
                        # Random flip
                        if np.random.random() > 0.5:
                            batch_X[i] = cv2.flip(batch_X[i], 1)
                            
                            # Handle mask shape for flip
                            if len(batch_y[i].shape) == 3:
                                mask_2d = np.squeeze(batch_y[i], axis=-1)
                                mask_flipped = cv2.flip(mask_2d, 1)
                                batch_y[i] = np.expand_dims(mask_flipped, axis=-1)
                            else:
                                batch_y[i] = cv2.flip(batch_y[i], 1)
                        
                        # Brightness and contrast augmentation (only for images, not masks)
                        if np.random.random() > 0.5:
                            # Brightness adjustment
                            brightness_factor = np.random.uniform(0.8, 1.2)
                            batch_X[i] = np.clip(batch_X[i] * brightness_factor, 0, 1)
                        
                        if np.random.random() > 0.5:
                            # Contrast adjustment
                            contrast_factor = np.random.uniform(0.8, 1.2)
                            mean = np.mean(batch_X[i])
                            batch_X[i] = np.clip((batch_X[i] - mean) * contrast_factor + mean, 0, 1)
                
                yield batch_X, batch_y
        
        train_gen = segmentation_generator(X_train, y_train, batch_size, augment=True)
        val_gen = segmentation_generator(X_val, y_val, batch_size, augment=False)
        
        return train_gen, val_gen
    
    def get_class_distribution(self, labels):
        """
        Get class distribution for analysis
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip([self.class_names[i] for i in unique], counts))
        return distribution

def load_test_data(data_loader, test_images_dir, test_masks_dir=None):
    """
    Load test data for evaluation
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading test data...")
    
    test_images = []
    test_masks = [] if test_masks_dir else None
    
    # Load test images (handle both flat directory and subdirectory structure)
    if os.path.isdir(test_images_dir):
        # Check if it's a flat directory or has subdirectories
        subdirs = [d for d in os.listdir(test_images_dir) 
                  if os.path.isdir(os.path.join(test_images_dir, d))]
        
        if subdirs:
            # Load from subdirectories (classification task structure)
            for subdir in subdirs:
                subdir_path = os.path.join(test_images_dir, subdir)
                for img_file in os.listdir(subdir_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(subdir_path, img_file)
                        try:
                            image = cv2.imread(img_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, (data_loader.img_width, data_loader.img_height))
                            image = image.astype(np.float32) / 255.0
                            test_images.append(image)
                        except Exception as e:
                            logger.error(f"Error loading test image {img_path}: {e}")
                            continue
        else:
            # Load from flat directory (segmentation task structure)
            for img_file in os.listdir(test_images_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(test_images_dir, img_file)
                    try:
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (data_loader.img_width, data_loader.img_height))
                        image = image.astype(np.float32) / 255.0
                        test_images.append(image)
                    except Exception as e:
                        logger.error(f"Error loading test image {img_path}: {e}")
                        continue
    
    test_images = np.array(test_images)
    logger.info(f"Loaded {len(test_images)} test images")
    
    # Load test masks if provided
    if test_masks_dir:
        for mask_file in os.listdir(test_masks_dir):
            if mask_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                mask_path = os.path.join(test_masks_dir, mask_file)
                try:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (data_loader.img_width, data_loader.img_height))
                    mask = (mask > 128).astype(np.float32)
                    test_masks.append(mask)
                except Exception as e:
                    logger.error(f"Error loading test mask {mask_path}: {e}")
                    continue
        
        test_masks = np.array(test_masks)
        test_masks = np.expand_dims(test_masks, axis=-1)
        logger.info(f"Loaded {len(test_masks)} test masks")
    
    return test_images, test_masks
