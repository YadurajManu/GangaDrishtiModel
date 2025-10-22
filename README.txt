================================================================================
                    MICROPLASTIC DETECTION USING YOLO 11
================================================================================

PROJECT OVERVIEW:
This project implements a computer vision model for detecting microplastic particles
in environmental samples using YOLO 11 (Ultralytics) with polygon annotations.

DATASET STRUCTURE:
├── train/
│   ├── images/          # 36 training images (.jpg)
│   └── labels/          # 36 corresponding label files (.txt)
├── valid/
│   ├── images/          # 10 validation images (.jpg)
│   └── labels/          # 10 corresponding label files (.txt)
└── test/
    ├── images/          # 5 test images (.jpg)
    └── labels/          # 5 corresponding label files (.txt)

TOTAL DATASET: 51 images with polygon annotations

ANNOTATION FORMAT:
- YOLO format with polygon coordinates
- Each line: class_id x1 y1 x2 y2 x3 y3 ... (normalized 0-1)
- Multiple classes detected (0, 1, 2, 3+)
- Polygon shapes capture irregular microplastic particle boundaries

CLASSES DETECTED:
- Class 0: [To be defined - analyze dataset]
- Class 1: [To be defined - analyze dataset]
- Class 2: [To be defined - analyze dataset]
- Class 3: [To be defined - analyze dataset]

REQUIREMENTS:
- Python 3.8+
- PyTorch
- Ultralytics YOLO 11
- OpenCV
- NumPy
- Matplotlib
- Pillow

INSTALLATION STEPS:
1. Install Python dependencies:
   pip install ultralytics torch torchvision opencv-python numpy matplotlib pillow

2. Verify YOLO 11 installation:
   python -c "from ultralytics import YOLO; print('YOLO 11 installed successfully')"

TRAINING CONFIGURATION:
- Model: YOLOv11n (nano) for faster training on small dataset
- Input size: 640x640 pixels
- Epochs: 100-200 (adjust based on convergence)
- Batch size: 8-16 (adjust based on GPU memory)
- Learning rate: 0.01 (default)
- Data augmentation: Enabled for small dataset

EXPECTED PERFORMANCE:
- Small dataset (51 images) may require:
  * Transfer learning from pre-trained weights
  * Data augmentation techniques
  * Careful validation to avoid overfitting

NEXT STEPS:
1. Set up YOLO 11 environment
2. Create dataset configuration file
3. Implement training script
4. Add data augmentation
5. Train and evaluate model
6. Test on new microplastic samples

NOTES:
- This is a specialized dataset for environmental microplastic detection
- Polygon annotations provide precise particle boundary detection
- Consider collecting more data for improved model performance
- Regular validation monitoring recommended due to small dataset size

================================================================================
CREATED: [Current Date]
AUTHOR: [Your Name]
PROJECT: Microplastic Detection using YOLO 11
================================================================================
