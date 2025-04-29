# Motorcycle Helmet Detection Project

## Project Overview
This project implements and evaluates different object detection models for motorcycle helmet detection. The goal is to accurately detect and count helmets in images of motorcycles and riders, which is crucial for road safety monitoring and enforcement.

## Models Evaluated
The project evaluates and compares the performance of several state-of-the-art object detection models:

- YOLOv5 (nano)
- YOLOv8 (nano)
- YOLOv9c
- YOLOv12 (nano)
- RT-DETR (large)

Each model was trained using 5-fold cross-validation to ensure robust evaluation.

## Dataset Information
The dataset contains images of motorcycles with riders, annotated with three classes:
- Class 0: helmet
- Class 1: driver
- Class 2: passenger

Images are organized in the following structure:
- Training images in the `Data/images` directory
- Labels in YOLO format in the `Data/labels` directory
- Test dataset in `test_dataset` and `test_combined_output` directories

## Project Structure
```
.
├── ALL_output/               # Combined model evaluation results
├── Final_Code/               # Final implementation of models
│   ├── code/                 # Python scripts for training and evaluation
│   ├── config/               # YAML configuration files
│   └── evaluation_results/   # Detailed model performance reports
├── code/                     # Development scripts
│   └── fold_*/              # Cross-validation fold datasets
├── Data/                     # Main dataset directory
└── backupResult/             # Backup of model results
```

## Key Features
- Multi-model comparison (YOLOv5, YOLOv8, YOLOv9c, YOLOv12, RT-DETR)
- 5-fold cross-validation for robust model evaluation
- Performance metrics for both classification (helmet presence) and count accuracy
- Visualization tools for model comparison (confusion matrices, charts)

## Performance Metrics
Models are evaluated on several metrics:

### Classification Metrics (Helmet Presence)
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Count Metrics (Number of Helmets)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Count Accuracy (Exact Match)

## Model Comparison Summary
Based on the evaluation results:

### Best Models by Category
- **Best Classification Performance (helmet presence detection)**: RT-DETR with F1 Score of 0.5683
- **Best Count Accuracy (helmet count)**: RT-DETR with MAE of 0.7161
- **Overall Best Model**: RT-DETR provides the best balance of metrics

### Detailed Performance Comparison
| Model   | Accuracy | Precision | Recall | F1 Score | MAE    | RMSE   |
|---------|----------|-----------|--------|----------|--------|--------|
| YOLOv5  | 0.2988   | 0.9752    | 0.0960 | 0.1748   | 0.9506 | 1.1952 |
| YOLOv8  | 0.4408   | 0.9756    | 0.2843 | 0.4402   | 0.8008 | 1.0859 |
| YOLOv12 | 0.3009   | 0.9903    | 0.0973 | 0.1772   | 0.9489 | 1.1950 |
| RT-DETR | 0.5275   | 0.9688    | 0.4021 | 0.5683   | 0.7161 | 1.0291 |

## Usage

### Model Training
To train a model using a specific configuration:

```bash
python code/main.py --yaml_path path/to/config.yaml
```

### Model Evaluation
To evaluate models:

```bash
python Final_Code/code/evaluation.py --model-type Yolov8 RTDETR --ground-truth path/to/ground_truth.csv --images path/to/test/images --conf 0.25
```

### Prediction
To make predictions on new images:

```bash
python Final_Code/code/prediction.py --model path/to/model.pt --images path/to/images --output output_folder_name --conf 0.25 --save-annotated
```

## Key Files
- `Final_Code/code/evaluation.py`: Comprehensive evaluation script for all models
- `Final_Code/code/prediction.py`: Script for making predictions on new images
- `Final_Code/config/helmet.yaml`: Configuration template for model training
- `ALL_output/Helmet_YOLOv9c_summary.txt`: Summary of YOLOv9c performance

## Dependencies
- Python 3.10+
- PyTorch
- Ultralytics YOLO/RTDETR
- OpenCV
- scikit-learn
- Matplotlib
- NumPy
- Pandas