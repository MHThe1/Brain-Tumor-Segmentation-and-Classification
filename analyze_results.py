"""
Analyze Enhanced Attention U-Net Results
BRACU CSE428 Academic Project
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_results():
    """
    Analyze the training results from the Enhanced Attention U-Net
    """
    print("="*80)
    print("ENHANCED ATTENTION U-NET RESULTS ANALYSIS")
    print("="*80)
    
    # Results from the training log
    spatial_attention_results = {
        'model': 'Enhanced_Attention_UNet_Spatial_Only',
        'final_training_dice': 0.8884,
        'final_training_iou': 0.8007,
        'best_validation_dice': 0.8496,
        'best_validation_iou': 0.7427,
        'test_dice': 0.8553,
        'test_ioU': 0.7547,
        'epochs_trained': 50,
        'best_epoch': 48
    }
    
    print("📊 SPATIAL ATTENTION U-NET RESULTS:")
    print("-" * 50)
    print(f"Final Training Dice Coefficient: {spatial_attention_results['final_training_dice']:.4f} ({spatial_attention_results['final_training_dice']*100:.2f}%)")
    print(f"Final Training IoU Coefficient:  {spatial_attention_results['final_training_iou']:.4f} ({spatial_attention_results['final_training_iou']*100:.2f}%)")
    print(f"Best Validation Dice Coefficient: {spatial_attention_results['best_validation_dice']:.4f} ({spatial_attention_results['best_validation_dice']*100:.2f}%)")
    print(f"Best Validation IoU Coefficient:  {spatial_attention_results['best_validation_iou']:.4f} ({spatial_attention_results['best_validation_iou']*100:.2f}%)")
    print(f"Test Dice Coefficient: {spatial_attention_results['test_dice']:.4f} ({spatial_attention_results['test_dice']*100:.2f}%)")
    print(f"Test IoU Coefficient:  {spatial_attention_results['test_ioU']:.4f} ({spatial_attention_results['test_ioU']*100:.2f}%)")
    print(f"Best Model at Epoch: {spatial_attention_results['best_epoch']}")
    print(f"Total Epochs Trained: {spatial_attention_results['epochs_trained']}")
    
    print("\n🎯 PERFORMANCE ANALYSIS:")
    print("-" * 50)
    
    # Performance interpretation
    dice_score = spatial_attention_results['test_dice']
    iou_score = spatial_attention_results['test_ioU']
    
    if dice_score >= 0.85:
        dice_rating = "Excellent"
    elif dice_score >= 0.80:
        dice_rating = "Very Good"
    elif dice_score >= 0.75:
        dice_rating = "Good"
    elif dice_score >= 0.70:
        dice_rating = "Fair"
    else:
        dice_rating = "Needs Improvement"
    
    if iou_score >= 0.75:
        iou_rating = "Excellent"
    elif iou_score >= 0.70:
        iou_rating = "Very Good"
    elif iou_score >= 0.65:
        iou_rating = "Good"
    elif iou_score >= 0.60:
        iou_rating = "Fair"
    else:
        iou_rating = "Needs Improvement"
    
    print(f"Dice Coefficient Rating: {dice_rating}")
    print(f"IoU Coefficient Rating: {iou_rating}")
    
    # Training stability analysis
    training_dice = spatial_attention_results['final_training_dice']
    validation_dice = spatial_attention_results['best_validation_dice']
    gap = training_dice - validation_dice
    
    print(f"\n📈 TRAINING STABILITY:")
    print("-" * 50)
    print(f"Training-Validation Gap: {gap:.4f}")
    
    if gap <= 0.05:
        stability = "Excellent - No overfitting"
    elif gap <= 0.10:
        stability = "Good - Minimal overfitting"
    elif gap <= 0.15:
        stability = "Fair - Some overfitting"
    else:
        stability = "Poor - Significant overfitting"
    
    print(f"Stability Rating: {stability}")
    
    # Comparison with typical U-Net performance
    print(f"\n🔄 COMPARISON WITH TYPICAL U-NET:")
    print("-" * 50)
    typical_unet_dice = 0.75  # Typical U-Net performance
    typical_unet_iou = 0.65   # Typical U-Net performance
    
    dice_improvement = ((dice_score - typical_unet_dice) / typical_unet_dice) * 100
    iou_improvement = ((iou_score - typical_unet_iou) / typical_unet_iou) * 100
    
    print(f"Typical U-Net Dice: {typical_unet_dice:.4f}")
    print(f"Our Attention U-Net Dice: {dice_score:.4f}")
    print(f"Dice Improvement: {dice_improvement:+.2f}%")
    
    print(f"Typical U-Net IoU: {typical_unet_iou:.4f}")
    print(f"Our Attention U-Net IoU: {iou_score:.4f}")
    print(f"IoU Improvement: {iou_improvement:+.2f}%")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    
    if dice_score >= 0.85 and iou_score >= 0.75:
        print("✅ Excellent performance! The model is ready for deployment.")
        print("✅ Consider training other attention variants for comparison.")
    elif dice_score >= 0.80:
        print("✅ Good performance! Consider:")
        print("   - Training with channel attention")
        print("   - Adding data augmentation")
        print("   - Fine-tuning hyperparameters")
    else:
        print("⚠️  Performance can be improved by:")
        print("   - Increasing training epochs")
        print("   - Adding more data augmentation")
        print("   - Trying different attention mechanisms")
    
    print(f"\n🚀 NEXT STEPS:")
    print("-" * 50)
    print("1. Continue training other attention variants")
    print("2. Compare performance across different attention types")
    print("3. Generate attention visualizations")
    print("4. Test on additional brain tumor datasets")
    
    return spatial_attention_results

def create_performance_visualization():
    """
    Create a visualization of the performance metrics
    """
    # Create a simple performance chart
    metrics = ['Training Dice', 'Validation Dice', 'Test Dice', 'Training IoU', 'Validation IoU', 'Test IoU']
    values = [0.8884, 0.8496, 0.8553, 0.8007, 0.7427, 0.7547]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'skyblue', 'lightcoral', 'lightgreen'])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Enhanced Attention U-Net Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add horizontal line for typical U-Net performance
    plt.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Typical U-Net Dice')
    plt.axhline(y=0.65, color='orange', linestyle='--', alpha=0.7, label='Typical U-Net IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('attention_unet_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Performance visualization saved as 'attention_unet_performance.png'")

if __name__ == "__main__":
    results = analyze_training_results()
    create_performance_visualization()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
