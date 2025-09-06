#!/usr/bin/env python3
"""
Brain Tumor Inference Script
BRACU CSE428 Academic Project
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def check_gpu_status():
    """Check and display GPU status"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU available: {gpus[0].name}")
        return True
    else:
        print("⚠️  No GPU detected, using CPU")
        return False

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image for inference"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        
        return image_batch, image
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None, None

def run_inference_on_image(image_path, model_type="simple_cnn"):
    """Run inference on a single image"""
    print_header(f"Inference on {os.path.basename(image_path)}")
    
    # Check GPU
    gpu_available = check_gpu_status()
    
    # Load and preprocess image
    print(f"📸 Loading image: {image_path}")
    image_batch, original_image = load_and_preprocess_image(image_path)
    
    if image_batch is None:
        return
    
    print(f"✅ Image loaded and preprocessed: {image_batch.shape}")
    
    # Load model (you'll need to have trained models available)
    model_path = f"outputs/run_*/models/{model_type}_classification_best.h5"
    
    try:
        # Find the latest model
        import glob
        model_files = glob.glob(model_path)
        if not model_files:
            print(f"❌ No trained model found for {model_type}")
            print("Please train a model first using run_training.py")
            return
        
        latest_model = max(model_files, key=os.path.getctime)
        print(f"📁 Loading model: {latest_model}")
        
        # Load the model
        model = tf.keras.models.load_model(latest_model)
        
        # Run inference
        print("🔮 Running inference...")
        predictions = model.predict(image_batch, verbose=0)
        
        # Get class names
        class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        print("\n" + "="*50)
        print("🎯 PREDICTION RESULTS")
        print("="*50)
        print(f"📊 Predicted class: {predicted_class}")
        print(f"🎯 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("\n📈 All class probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
            print(f"   {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        # Display image with prediction
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f"Input Image\nPredicted: {predicted_class}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(class_names, predictions[0])
        plt.title("Class Probabilities")
        plt.ylabel("Probability")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence, predictions[0]
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None, None, None

def run_batch_inference(image_dir, model_type="simple_cnn"):
    """Run inference on multiple images"""
    print_header(f"Batch Inference on {image_dir}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"📸 Found {len(image_files)} images")
    
    # Load model once
    model_path = f"outputs/run_*/models/{model_type}_classification_best.h5"
    model_files = glob.glob(model_path)
    if not model_files:
        print(f"❌ No trained model found for {model_type}")
        return
    
    latest_model = max(model_files, key=os.path.getctime)
    model = tf.keras.models.load_model(latest_model)
    class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    
    results = []
    
    for i, image_path in enumerate(image_files[:10]):  # Limit to first 10 images
        print(f"\n📸 Processing {i+1}/{min(10, len(image_files))}: {os.path.basename(image_path)}")
        
        image_batch, _ = load_and_preprocess_image(image_path)
        if image_batch is None:
            continue
        
        predictions = model.predict(image_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        results.append({
            'image': os.path.basename(image_path),
            'predicted_class': predicted_class,
            'confidence': confidence
        })
        
        print(f"   → {predicted_class} ({confidence:.3f})")
    
    # Summary
    print("\n" + "="*50)
    print("📊 BATCH INFERENCE SUMMARY")
    print("="*50)
    for result in results:
        print(f"{result['image']:30} → {result['predicted_class']:12} ({result['confidence']:.3f})")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Brain Tumor Inference Script")
    parser.add_argument("--image", type=str, help="Path to single image for inference")
    parser.add_argument("--batch", type=str, help="Path to directory of images for batch inference")
    parser.add_argument("--model", type=str, default="simple_cnn", 
                       choices=["simple_cnn", "resnet50", "efficientnet", "densenet"],
                       help="Model type to use for inference")
    
    args = parser.parse_args()
    
    print_header("Brain Tumor Inference Pipeline")
    print("BRACU CSE428 Academic Project")
    
    if args.image:
        run_inference_on_image(args.image, args.model)
    elif args.batch:
        run_batch_inference(args.batch, args.model)
    else:
        print("❌ Please specify either --image or --batch")
        print("\nUsage examples:")
        print("  python run_inference.py --image path/to/brain_mri.jpg")
        print("  python run_inference.py --batch path/to/images/")
        print("  python run_inference.py --image path/to/brain_mri.jpg --model resnet50")

if __name__ == "__main__":
    main()
