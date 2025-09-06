"""
Utility functions for Brain Tumor Segmentation and Classification Project
BRACU CSE428 Academic Project
"""

import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image

def setup_logging(log_file_path, log_level="INFO"):
    """
    Set up logging configuration for the project
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_experiment_config(config_dict, output_dir):
    """
    Save experiment configuration to JSON file
    """
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)
    print(f"Experiment configuration saved to: {config_path}")

def plot_training_history(history, output_dir, model_name):
    """
    Plot and save training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - {model_name}', fontsize=16)
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'])
    axes[0, 0].plot(history.history['val_accuracy'])
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'])
    axes[0, 1].plot(history.history['val_loss'])
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend(['Train', 'Validation'], loc='upper left')
    
    # Plot segmentation metrics if available
    if 'segmentation_loss' in history.history:
        axes[1, 0].plot(history.history['segmentation_loss'])
        axes[1, 0].plot(history.history['val_segmentation_loss'])
        axes[1, 0].set_title('Segmentation Loss')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend(['Train', 'Validation'], loc='upper left')
    
    if 'segmentation_dice_coef' in history.history:
        axes[1, 1].plot(history.history['segmentation_dice_coef'])
        axes[1, 1].plot(history.history['val_segmentation_dice_coef'])
        axes[1, 1].set_title('Segmentation Dice Coefficient')
        axes[1, 1].set_ylabel('Dice Coefficient')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plots", f"{model_name}_training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {plot_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, model_name):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = os.path.join(output_dir, "plots", f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to: {plot_path}")

def save_classification_report(y_true, y_pred, class_names, output_dir, model_name):
    """
    Save classification report to file
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, "results", f"{model_name}_classification_report.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save as text
    txt_path = os.path.join(output_dir, "results", f"{model_name}_classification_report.txt")
    with open(txt_path, 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"Classification report saved to: {json_path} and {txt_path}")

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient for segmentation evaluation
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for segmentation evaluation
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess image for model input
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image

def visualize_prediction(image, mask_true, mask_pred, output_path, title="Prediction Visualization"):
    """
    Visualize segmentation prediction with original image, true mask, and predicted mask
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_true, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    axes[2].imshow(mask_pred, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction visualization saved to: {output_path}")

def save_model_summary(model, output_dir, model_name):
    """
    Save model summary to file
    """
    summary_path = os.path.join(output_dir, "models", f"{model_name}_summary.txt")
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to: {summary_path}")

def log_experiment_results(results_dict, output_dir, experiment_name):
    """
    Log experiment results to JSON file
    """
    results_path = os.path.join(output_dir, "results", f"{experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4, default=str)
    print(f"Experiment results saved to: {results_path}")

def create_experiment_report(experiment_name, config, results, output_dir):
    """
    Create a comprehensive experiment report
    """
    report_path = os.path.join(output_dir, "results", f"{experiment_name}_report.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Experiment Report: {experiment_name}\n\n")
        f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n")
        f.write("```json\n")
        f.write(json.dumps(config, indent=4, default=str))
        f.write("\n```\n\n")
        
        f.write("## Results\n")
        f.write("```json\n")
        f.write(json.dumps(results, indent=4, default=str))
        f.write("\n```\n\n")
        
        f.write("## Analysis\n")
        f.write("### Key Findings\n")
        f.write("- [Add your analysis here]\n\n")
        f.write("### Recommendations\n")
        f.write("- [Add your recommendations here]\n\n")
    
    print(f"Experiment report saved to: {report_path}")

def print_experiment_header(experiment_name, description=""):
    """
    Print a formatted header for experiments
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    if description:
        print(f"DESCRIPTION: {description}")
    print(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
