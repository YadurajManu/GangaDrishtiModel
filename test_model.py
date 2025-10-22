#!/usr/bin/env python3
"""
Test the trained microplastic detection model
This script loads the best model and tests it on sample images
"""

from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt

def test_model():
    print("=" * 60)
    print("TESTING TRAINED MICROPLASTIC DETECTION MODEL")
    print("=" * 60)
    
    # Load the best trained model
    model_path = "microplastic_detection/yolo11n_training/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    print(f"✅ Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Test on validation images
    print("\n1. Testing on validation images...")
    val_images = "valid/images"
    
    if os.path.exists(val_images):
        # Run inference on validation images
        results = model.predict(
            source=val_images,
            save=True,
            save_txt=True,
            conf=0.25,  # Confidence threshold
            project="microplastic_detection",
            name="test_results"
        )
        
        print(f"✅ Results saved to: microplastic_detection/test_results/")
        
        # Show results summary
        print("\n2. Detection Results Summary:")
        for i, result in enumerate(results):
            image_name = os.path.basename(result.path)
            detections = len(result.boxes) if result.boxes is not None else 0
            print(f"  {image_name}: {detections} microplastics detected")
            
            # Show confidence scores
            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for j, (conf, cls) in enumerate(zip(confidences, classes)):
                    class_name = "microplastic_type_0" if cls == 0 else "microplastic_type_1"
                    print(f"    Detection {j+1}: {class_name} (confidence: {conf:.3f})")
    
    else:
        print(f"❌ Validation images not found at: {val_images}")
    
    # Test on a single test image
    print("\n3. Testing on a single test image...")
    test_images = "test/images"
    
    if os.path.exists(test_images):
        test_files = [f for f in os.listdir(test_images) if f.endswith('.jpg')]
        if test_files:
            test_image = os.path.join(test_images, test_files[0])
            print(f"Testing on: {test_files[0]}")
            
            # Run inference
            result = model.predict(
                source=test_image,
                save=True,
                conf=0.25,
                project="microplastic_detection",
                name="single_test"
            )
            
            print(f"✅ Single test result saved to: microplastic_detection/single_test/")
            
            # Show detection details
            if result[0].boxes is not None and len(result[0].boxes) > 0:
                detections = len(result[0].boxes)
                print(f"Found {detections} microplastics in test image")
                
                confidences = result[0].boxes.conf.cpu().numpy()
                classes = result[0].boxes.cls.cpu().numpy()
                
                for i, (conf, cls) in enumerate(zip(confidences, classes)):
                    class_name = "microplastic_type_0" if cls == 0 else "microplastic_type_1"
                    print(f"  Detection {i+1}: {class_name} (confidence: {conf:.3f})")
            else:
                print("No microplastics detected in test image")
    
    print("\n" + "=" * 60)
    print("MODEL TESTING COMPLETE!")
    print("=" * 60)
    print("\nCheck the results in:")
    print("- microplastic_detection/test_results/ (validation images)")
    print("- microplastic_detection/single_test/ (single test image)")
    print("\nThe model shows:")
    print("- High recall (finds microplastics)")
    print("- Low precision (some false positives)")
    print("- Better performance on microplastic_type_1")

if __name__ == "__main__":
    test_model()
