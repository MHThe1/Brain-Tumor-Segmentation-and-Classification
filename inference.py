"""
Inference script for Brain Tumor Segmentation and Classification
BRACU CSE428 Academic Project

This script provides functionality to:
1. Load a single image
2. Run segmentation and classification
3. Display results in the required format
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import argparse
import logging

# Import project modules
from config import *
from utils import *
from models import UNet, AttentionUNet, MultiTaskUNet, ClassifierModels

class BrainTumorInference:
    """
    Inference class for brain tumor models
    """
    
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or os.path.join(PROJECT_ROOT, "outputs")
        self.logger = setup_logging(
            os.path.join(RUN_OUTPUT_DIR, "logs", "inference.log"),
            LOG_LEVEL
        )
        
        # Initialize models (will be loaded when needed)
        self.segmentation_model = None
        self.classification_model = None
        self.multitask_model = None
        
        # Class names
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    
    def load_segmentation_model(self, model_path, model_type="unet"):
        """
        Load segmentation model
        """
        self.logger.info(f"Loading {model_type} segmentation model from {model_path}")
        
        if model_type == "unet":
            model = UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                        num_classes=SEGMENTATION_CLASSES)
            self.segmentation_model = model.build_model()
        elif model_type == "attention_unet":
            model = AttentionUNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                                num_classes=SEGMENTATION_CLASSES)
            self.segmentation_model = model.build_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        self.segmentation_model.load_weights(model_path)
        self.logger.info("Segmentation model loaded successfully")
    
    def load_classification_model(self, model_path, model_type="simple_cnn"):
        """
        Load classification model
        """
        self.logger.info(f"Loading {model_type} classification model from {model_path}")
        
        if model_type == "simple_cnn":
            self.classification_model = ClassifierModels.simple_cnn(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                num_classes=NUM_CLASSES
            )
        elif model_type == "resnet50":
            self.classification_model = ClassifierModels.resnet50_classifier(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                num_classes=NUM_CLASSES
            )
        elif model_type == "efficientnet":
            self.classification_model = ClassifierModels.efficientnet_classifier(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                num_classes=NUM_CLASSES
            )
        elif model_type == "densenet":
            self.classification_model = ClassifierModels.densenet_classifier(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
                num_classes=NUM_CLASSES
            )
        else:
            raise ValueError(f"Unknown classifier type: {model_type}")
        
        # Load weights
        self.classification_model.load_weights(model_path)
        self.logger.info("Classification model loaded successfully")
    
    def load_multitask_model(self, model_path):
        """
        Load multi-task model
        """
        self.logger.info(f"Loading multi-task model from {model_path}")
        
        model = MultiTaskUNet(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            seg_classes=SEGMENTATION_CLASSES,
            cls_classes=NUM_CLASSES
        )
        self.multitask_model = model.build_model()
        
        # Load weights
        self.multitask_model.load_weights(model_path)
        self.logger.info("Multi-task model loaded successfully")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, image_resized
    
    def predict_segmentation(self, image_batch):
        """
        Predict segmentation mask
        """
        if self.segmentation_model is None:
            raise ValueError("Segmentation model not loaded")
        
        prediction = self.segmentation_model.predict(image_batch, verbose=0)
        mask = prediction[0, :, :, 0]  # Get first channel
        
        # Threshold to get binary mask
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        return binary_mask, mask
    
    def predict_classification(self, image_batch):
        """
        Predict tumor classification
        """
        if self.classification_model is None:
            raise ValueError("Classification model not loaded")
        
        prediction = self.classification_model.predict(image_batch, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        class_name = self.class_names[class_idx]
        
        return class_name, confidence, prediction[0]
    
    def predict_multitask(self, image_batch):
        """
        Predict both segmentation and classification
        """
        if self.multitask_model is None:
            raise ValueError("Multi-task model not loaded")
        
        predictions = self.multitask_model.predict(image_batch, verbose=0)
        
        # Segmentation prediction
        seg_pred = predictions[0]
        mask = seg_pred[0, :, :, 0]
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Classification prediction
        cls_pred = predictions[1]
        class_idx = np.argmax(cls_pred[0])
        confidence = cls_pred[0][class_idx]
        class_name = self.class_names[class_idx]
        
        return binary_mask, mask, class_name, confidence, cls_pred[0]
    
    def visualize_results(self, original_image, mask, class_name, confidence, 
                         output_path=None, show_plot=True):
        """
        Visualize prediction results in the required format
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Brain MRI Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(original_image, cmap='gray', alpha=0.7)
        axes[1].imshow(mask, cmap='Reds', alpha=0.5)
        axes[1].set_title('Tumor Segmentation', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Classification result
        axes[2].text(0.5, 0.7, f'Tumor Type: {class_name.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', ha='center', va='center',
                    transform=axes[2].transAxes)
        axes[2].text(0.5, 0.5, f'Confidence: {confidence:.2%}', 
                    fontsize=14, ha='center', va='center',
                    transform=axes[2].transAxes)
        axes[2].text(0.5, 0.3, f'Status: {"Tumor Detected" if class_name != "no_tumor" else "No Tumor"}', 
                    fontsize=14, ha='center', va='center',
                    transform=axes[2].transAxes)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.suptitle('Brain Tumor Analysis Results', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Results visualization saved to: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def analyze_single_image(self, image_path, model_type="multitask", 
                           output_dir=None, show_plot=True):
        """
        Analyze a single brain MRI image
        """
        self.logger.info(f"Analyzing image: {image_path}")
        
        # Preprocess image
        image_batch, original_image = self.preprocess_image(image_path)
        
        # Make predictions based on model type
        if model_type == "multitask":
            if self.multitask_model is None:
                raise ValueError("Multi-task model not loaded. Please load a model first.")
            
            binary_mask, mask, class_name, confidence, class_probs = self.predict_multitask(image_batch)
            
        elif model_type == "separate":
            if self.segmentation_model is None or self.classification_model is None:
                raise ValueError("Both segmentation and classification models must be loaded.")
            
            binary_mask, mask = self.predict_segmentation(image_batch)
            class_name, confidence, class_probs = self.predict_classification(image_batch)
            
        else:
            raise ValueError("model_type must be 'multitask' or 'separate'")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"analysis_{os.path.basename(image_path)}.png")
        else:
            output_path = None
        
        # Visualize results
        self.visualize_results(original_image, binary_mask, class_name, confidence, 
                             output_path, show_plot)
        
        # Print results
        print(f"\n{'='*60}")
        print("BRAIN TUMOR ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Tumor Type: {class_name.replace('_', ' ').title()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Status: {'Tumor Detected' if class_name != 'no_tumor' else 'No Tumor'}")
        print(f"{'='*60}")
        
        # Save detailed results
        results = {
            'image_path': image_path,
            'tumor_type': class_name,
            'confidence': float(confidence),
            'class_probabilities': {
                name: float(prob) for name, prob in zip(self.class_names, class_probs)
            },
            'tumor_detected': class_name != 'no_tumor'
        }
        
        if output_dir:
            import json
            results_path = os.path.join(output_dir, f"results_{os.path.basename(image_path)}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            self.logger.info(f"Detailed results saved to: {results_path}")
        
        return results

def main():
    """
    Main inference function
    """
    parser = argparse.ArgumentParser(description='Brain Tumor Analysis Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to brain MRI image')
    parser.add_argument('--model_type', type=str, default='multitask', 
                       choices=['multitask', 'separate'], help='Model type to use')
    parser.add_argument('--seg_model', type=str, help='Path to segmentation model')
    parser.add_argument('--cls_model', type=str, help='Path to classification model')
    parser.add_argument('--multitask_model', type=str, help='Path to multi-task model')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--no_show', action='store_true', help='Do not show plot')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = BrainTumorInference()
    
    # Load models based on type
    if args.model_type == "multitask":
        if args.multitask_model:
            inference.load_multitask_model(args.multitask_model)
        else:
            # Try to find the latest multi-task model
            model_dir = os.path.join(PROJECT_ROOT, "outputs")
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if "multitask_unet_best.h5" in file:
                        inference.load_multitask_model(os.path.join(root, file))
                        break
    
    elif args.model_type == "separate":
        if args.seg_model and args.cls_model:
            inference.load_segmentation_model(args.seg_model)
            inference.load_classification_model(args.cls_model)
        else:
            print("Error: Both --seg_model and --cls_model must be provided for separate model type")
            return
    
    # Analyze image
    try:
        results = inference.analyze_single_image(
            args.image, 
            args.model_type, 
            args.output_dir, 
            not args.no_show
        )
        
        print(f"\nAnalysis completed successfully!")
        if args.output_dir:
            print(f"Results saved to: {args.output_dir}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
