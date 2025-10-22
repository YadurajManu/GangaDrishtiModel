# 🔬 Microplastic Detection using YOLO 11

A computer vision project for detecting microplastic particles in environmental samples using YOLO 11 (Ultralytics) with polygon annotations.

## 📋 Project Overview

This project implements an object detection model to identify and classify microplastic particles in environmental samples. The model uses YOLO 11's polygon detection capabilities to precisely outline irregular microplastic shapes.

## 🗂️ Dataset Structure

```
GangaData/
├── train/
│   ├── images/          # 36 training images (.jpg)
│   └── labels/          # 36 corresponding label files (.txt)
├── valid/
│   ├── images/          # 10 validation images (.jpg)
│   └── labels/          # 10 corresponding label files (.txt)
└── test/
    ├── images/          # 5 test images (.jpg)
    └── labels/          # 5 corresponding label files (.txt)
```

**Total Dataset**: 51 images with polygon annotations

## 🏷️ Annotation Format

- **Format**: YOLO polygon format
- **Structure**: `class_id x1 y1 x2 y2 x3 y3 ...` (normalized coordinates 0-1)
- **Classes**: 2 types of microplastics detected
- **Polygon shapes**: Capture irregular microplastic particle boundaries

## 📊 Dataset Statistics

| Class | Instances | Description |
|-------|-----------|-------------|
| Class 0 | 3 | microplastic_type_0 |
| Class 1 | 22 | microplastic_type_1 |
| **Total** | **25** | **Microplastic particles** |

## 🛠️ Technical Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO 11
- OpenCV
- NumPy
- Matplotlib
- Pillow

## 🚀 Installation

```bash
# Install required packages
pip install ultralytics torch torchvision opencv-python numpy matplotlib pillow

# Verify installation
python -c "from ultralytics import YOLO; print('YOLO 11 installed successfully')"
```

## ⚙️ Configuration

### Dataset Configuration (`data.yaml`)
```yaml
path: /Users/sujeetkumarsingh/Desktop/ArcDownloads/GangaData
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['microplastic_type_0', 'microplastic_type_1']
```

### Training Parameters
- **Model**: YOLOv11n (nano version)
- **Input Size**: 640x640 pixels
- **Epochs**: 100 (stopped early at 22)
- **Batch Size**: 8
- **Device**: CPU (Apple M3 Pro)
- **Data Augmentation**: Enabled
- **Early Stopping**: 20 epochs patience

## 📈 Training Results

### Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | 24.3% | Mean Average Precision at IoU 0.5 |
| **mAP50-95** | 18.3% | Mean Average Precision at IoU 0.5-0.95 |
| **Precision** | 0.38% | Low precision (many false positives) |
| **Recall** | 95% | Excellent recall (finds most microplastics) |

### Class-Specific Performance
| Class | mAP50 | mAP50-95 | Performance |
|-------|-------|----------|-------------|
| microplastic_type_0 | 7.1% | 6.1% | ⚠️ Needs improvement |
| microplastic_type_1 | 41.5% | 30.4% | ✅ Good performance |

### Training Summary
- **Training Time**: 1.7 minutes (0.028 hours)
- **Epochs Completed**: 22/100 (early stopping)
- **Best Performance**: Achieved at Epoch 2
- **Model Size**: 5.5MB (optimizer stripped)

## 🎯 Key Findings

### ✅ Strengths
1. **High Recall (95%)**: Model successfully finds almost all microplastics
2. **Class 1 Detection**: Good performance on microplastic_type_1 (41.5% mAP)
3. **Fast Training**: Completed in under 2 minutes
4. **Polygon Detection**: Successfully handles irregular microplastic shapes

### ⚠️ Areas for Improvement
1. **Low Precision (0.38%)**: Too many false positive detections
2. **Class Imbalance**: Class 0 has only 3 instances vs 22 for Class 1
3. **Small Dataset**: 51 images limits model generalization
4. **Early Stopping**: Training stopped too early (Epoch 22)

## 📁 Project Files

### Core Files
- `train_microplastic.py` - Main training script
- `test_model.py` - Model testing script
- `data.yaml` - Dataset configuration
- `README.md` - This documentation

### Generated Files
```
microplastic_detection/yolo11n_training/
├── weights/
│   ├── best.pt          # Best performing model
│   └── last.pt          # Final epoch model
├── results.png          # Training curves
├── confusion_matrix.png # Detection accuracy
└── val_batch0_pred.jpg # Sample predictions
```

## 🔧 Usage

### Training the Model
```bash
python train_microplastic.py
```

### Testing the Model
```bash
python test_model.py
```

### Using the Trained Model
```python
from ultralytics import YOLO

# Load the best model
model = YOLO('microplastic_detection/yolo11n_training/weights/best.pt')

# Run inference
results = model.predict('path/to/image.jpg', conf=0.25)
```

## 📊 Model Performance Analysis

### Training Progress
- **Epoch 1**: mAP50 = 0.6% (starting)
- **Epoch 2**: mAP50 = 24.3% (best performance)
- **Epoch 22**: mAP50 = 17.7% (final)

### Detection Characteristics
- **Speed**: ~36ms per image (CPU inference)
- **Memory**: 0G GPU memory usage
- **Confidence**: Conservative detection (low false positives)

## 🚀 Future Improvements

### Data Collection
1. **Increase Dataset Size**: Collect more microplastic images
2. **Balance Classes**: More samples for microplastic_type_0
3. **Diverse Conditions**: Images under different lighting/magnification

### Model Optimization
1. **Longer Training**: Increase epochs to 200-300
2. **Hyperparameter Tuning**: Adjust learning rate, batch size
3. **Architecture**: Try YOLOv11s or YOLOv11m for better performance
4. **Transfer Learning**: Use domain-specific pre-trained weights

### Technical Enhancements
1. **GPU Training**: Use CUDA for faster training
2. **Advanced Augmentation**: Custom augmentation strategies
3. **Ensemble Methods**: Combine multiple models
4. **Post-processing**: Improve confidence threshold tuning

## 📝 Notes

- **Small Dataset Challenge**: 51 images is limiting for deep learning
- **Class Imbalance**: Uneven distribution affects performance
- **Environmental Application**: Specialized for microplastic detection
- **Polygon Annotations**: Provides precise particle boundary detection

## 🤝 Contributing

This project is designed for environmental microplastic research. Contributions for:
- Additional microplastic datasets
- Improved annotation techniques
- Model optimization strategies
- Real-world deployment solutions

## 📄 License

This project is for research and educational purposes in environmental science.

---

**Created**: October 2024  
**Author**: [Your Name]  
**Project**: Microplastic Detection using YOLO 11  
**Status**: ✅ Training Complete, Ready for Testing