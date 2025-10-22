#!/usr/bin/env python3
"""
Microplastic Detection Training Script using YOLO 11
This script trains a YOLO model to detect microplastic particles in environmental samples.
"""

from ultralytics import YOLO
import os
import torch

def main():
    print("=" * 60)
    print("MICROPLASTIC DETECTION TRAINING WITH YOLO 11")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLO model (using nano version for faster training on small dataset)
    print("\n1. Loading YOLOv11n model...")
    model = YOLO('yolo11n.pt')  # This will download the pre-trained weights
    
    # Verify dataset configuration
    print("\n2. Verifying dataset configuration...")
    if not os.path.exists('data.yaml'):
        print("❌ Error: data.yaml not found!")
        return
    
    print("✅ Dataset configuration found")
    
    # Training parameters
    print("\n3. Setting up training parameters...")
    training_args = {
        'data': 'data.yaml',           # Dataset configuration
        'epochs': 100,                 # Number of training epochs
        'imgsz': 640,                  # Input image size
        'batch': 8,                    # Batch size (adjust based on GPU memory)
        'device': device,              # Use GPU if available
        'project': 'microplastic_detection',  # Project name
        'name': 'yolo11n_training',    # Experiment name
        'save': True,                  # Save checkpoints
        'save_period': 10,             # Save every 10 epochs
        'cache': False,                # Don't cache images (small dataset)
        'augment': True,               # Enable data augmentation
        'patience': 20,                # Early stopping patience
        'verbose': True,               # Verbose output
    }
    
    print("Training parameters:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    
    # Start training
    print("\n4. Starting training...")
    print("This may take a while depending on your hardware...")
    
    try:
        # Train the model
        results = model.train(**training_args)
        
        print("\n✅ Training completed successfully!")
        print(f"Results saved in: microplastic_detection/yolo11n_training/")
        
        # Print training summary
        print("\nTraining Summary:")
        print(f"Best model: microplastic_detection/yolo11n_training/weights/best.pt")
        print(f"Last model: microplastic_detection/yolo11n_training/weights/last.pt")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Validate the trained model
    print("\n5. Validating the trained model...")
    try:
        val_results = model.val()
        print("✅ Validation completed!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the training results in microplastic_detection/yolo11n_training/")
    print("2. Use the best.pt model for inference")
    print("3. Test on new microplastic images")
    print("4. Consider data augmentation if performance is low")

if __name__ == "__main__":
    main()
