import os
import csv
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from ultralytics import YOLO, RTDETR
from tqdm import tqdm
random.seed(42)
    
    # Import additional requirements
import os
import csv
import cv2
def load_model(model_path, args):
    """
    Load the YOLO or RTDETR model from the specified path and set proper class names
    """
    try:
        print(f"Loading model from {model_path}...", flush=True)
        if 'RTDETR' in args.model_type or args.redtr:
            print("Loading YOLO model...", flush=True)
            model = RTDETR(model_path)
            print(f"Loaded RTDETR model successfully", flush=True)
            
            # # For RTDETR models, use class_names instead of names
            # if hasattr(model, 'model') and hasattr(model.model, 'class_names'):
            #     # Properly set RTDETR class names
            #     model.model.class_names = ['helmet', 'driver', 'passenger']
            #     print(f"Set class names for RTDETR model: {model.model.class_names}", flush=True)
            # else:
            #     print("Warning: Could not set class names for RTDETR model", flush=True)
        else:
            print("Loading YOLO model...", flush=True)  
            model = YOLO(model_path)
            print(f"Loaded YOLO model successfully", flush=True)
            
            # For YOLO models, use names
            # if hasattr(model, 'names'):
            #     model.names = {0: 'helmet', 1: 'driver', 2: 'passenger'}
            #     print(f"Set class names for YOLO model: {model.names}", flush=True)
            # elif hasattr(model, 'model') and hasattr(model.model, 'names'):
            #     model.model.names = {0: 'helmet', 1: 'driver', 2: 'passenger'}
            #     print(f"Set class names using model.model.names", flush=True)
            # else:
            #     print("Warning: Could not set class names for YOLO model", flush=True)
                
        return model
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def predict_helmet(image_path, model, conf_threshold=0.25):
    """
    Predict ONLY helmets in an image using the loaded model
    
    Args:
        image_path (str): Path to the image
        model: Loaded YOLO or RTDETR model
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        dict: Contains information about helmet presence and count
    """
    # Initialize result dictionary
    prediction_result = {
        "image_name": os.path.basename(image_path),
        "helmet_presence": 0,
        "helmet_count": 0
    }
    
    try:
        # Determine if this is RTDETR or YOLO
        is_rtdetr = isinstance(model, RTDETR)
        
        # Run prediction - RTDETR may need different parameters
        if is_rtdetr:
            # RTDETR specific parameters if needed
            results = model(image_path, conf=conf_threshold, verbose=False, classes=[0])[0]  # Only detect helmet class (0)
        else:
            # Standard YOLO parameters
            results = model(image_path, conf=conf_threshold, verbose=False, classes=[0])[0]  # Only detect helmet class (0)
        
        # Count helmets - all detections are helmets now since we filtered with classes=[0]
        helmet_detections = results.boxes.data.tolist()
        
        # Update result with helmet information
        helmet_count = len(helmet_detections)
        prediction_result["helmet_count"] = helmet_count
        prediction_result["helmet_presence"] = 1 if helmet_count > 0 else 0
        
        return prediction_result, results
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(image_path)}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return prediction_result, None

def create_output_directory(args):
    """
    Create a single output directory structure with timestamp for all models
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir_name = args.dir_name if args.dir_name else "Model_Evaluation"
    output_dir = os.path.join(base_dir_name, f"conf_{args.conf}_{timestamp}")
    
    # Create main directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each model type
    model_types = args.model_type if isinstance(args.model_type, list) else [args.model_type]
    for model_type in model_types:
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
    
    # Create a summary directory for comparative results
    summary_dir = os.path.join(output_dir, "Summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    print(f"Created output directory structure at: {output_dir}", flush=True)
    return output_dir

def save_annotated_image(results, output_dir, image_name):
    """
    Save annotated image with bounding boxes for helmets only
    
    Args:
        results: YOLO prediction results
        output_dir (str): Directory to save the image
        image_name (str): Original image name
    """
    # Get the original image
    orig_img = results.orig_img.copy()
    
    # All detections are helmets since we filtered with classes=[0] in predict_helmet
    helmet_detections = results.boxes.data.tolist()
    
    # If no helmet detections, skip saving this image
    if not helmet_detections:
        return
    
    # Draw bounding boxes for helmet detections
    for det in helmet_detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle (green color for helmet)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with confidence
        label = f"Helmet: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(orig_img, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width + 10, y1), 
                    (0, 255, 0), 
                    -1)
        
        # Draw text
        cv2.putText(orig_img, 
                   label, 
                   (x1 + 5, y1 - 5), 
                   font, 
                   font_scale, 
                   (0, 0, 0), 
                   thickness)
    
    # Save the image only if we drew helmet boxes
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, orig_img)

def load_ground_truth(csv_path):
    """
    Load ground truth data from CSV file
    
    Args:
        csv_path (str): Path to ground truth CSV
        
    Returns:
        dict: Dictionary with image_name as key and values as dict with presence and count
    """
    ground_truth = {}
    
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ground_truth[row['image_name']] = {
                    'helmet_presence': int(row['helmet_presence']),
                    'helmet_count': int(row['helmet_count'])
                }
        print(f"Loaded {len(ground_truth)} ground truth entries", flush=True)
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth CSV: {e}", flush=True)
        return {}

def calculate_metrics(ground_truth, predictions):
    """
    Calculate performance metrics by comparing predictions to ground truth
    
    Args:
        ground_truth (dict): Ground truth data
        predictions (list): List of prediction dictionaries
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Extract values for metrics calculation
    y_true_presence = []
    y_pred_presence = []
    y_true_count = []
    y_pred_count = []
    
    # Process only images that exist in ground truth
    valid_predictions = []
    
    for pred in tqdm(predictions, desc="Calculating metrics", unit="prediction"):
        image_name = pred['image_name']
        if image_name in ground_truth:
            y_true_presence.append(ground_truth[image_name]['helmet_presence'])
            y_pred_presence.append(pred['helmet_presence'])
            y_true_count.append(ground_truth[image_name]['helmet_count'])
            y_pred_count.append(pred['helmet_count'])
            valid_predictions.append(pred)
    
    # Return empty metrics if no valid predictions
    if not valid_predictions:
        print("No valid predictions found for metrics calculation", flush=True)
        return {}
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_true_presence, y_pred_presence)
    precision = precision_score(y_true_presence, y_pred_presence, zero_division=0)
    recall = recall_score(y_true_presence, y_pred_presence, zero_division=0)
    f1 = f1_score(y_true_presence, y_pred_presence, zero_division=0)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true_count, y_pred_count)
    mae = mean_absolute_error(y_true_count, y_pred_count)
    rmse = np.sqrt(mse)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'num_samples': len(valid_predictions)
    }
    
    return metrics

def process_images(image_paths, model_path, ground_truth, base_output_dir, model_type, fold_num, conf_threshold=0.25, save_annotated=False, args=None):
    """
    Process images and calculate metrics for a single model fold
    
    Args:
        image_paths (str or list): Directory containing images or list of image paths
        model_path (str): Path to the YOLO model
        ground_truth (dict): Ground truth data
        base_output_dir (str): Base output directory
        model_type (str): Model type name
        fold_num (int): Model fold number
        conf_threshold (float): Confidence threshold
        save_annotated (bool): Whether to save annotated images
        args: Command line arguments
        
    Returns:
        tuple: Metrics dictionary and list of predictions
    """
    # Load model
    model = load_model(model_path, args=args)
    if model is None:
        print(f"Failed to load model {model_path}, skipping this fold", flush=True)
        return None, None
    
    # Get model specific directory
    output_dir = os.path.join(base_output_dir, model_type)
    
    # Create subfolder for annotated images if needed
    images_dir = None
    if save_annotated:
        images_dir = os.path.join(output_dir, f"fold_{fold_num}_annotated_images")
        os.makedirs(images_dir, exist_ok=True)
    
    # Initialize results list for CSV
    predictions = []
    
    # Handle directory or list of image paths
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        # Find all image files
        image_paths = [os.path.join(image_paths, f) for f in os.listdir(image_paths)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Process each image
    for img_path in tqdm(image_paths, desc=f"Processing {model_type} fold {fold_num}", unit="image"):
        try:
            # Make prediction
            result, yolo_results = predict_helmet(img_path, model, conf_threshold)
            predictions.append(result)
            
            # Save annotated image if requested
            if save_annotated and yolo_results is not None:
                save_annotated_image(yolo_results, images_dir, os.path.basename(img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}", flush=True)
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)
    
    # Save predictions to CSV
    csv_path = os.path.join(output_dir, f"fold_{fold_num}_predictions.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'helmet_presence', 'helmet_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for pred in predictions:
            writer.writerow(pred)
    
    print(f"Predictions for {model_type} fold {fold_num} saved to {csv_path}", flush=True)
    
    return metrics, predictions

def evaluate_model_fold(model_type, fold, ground_truth_csv, images_path, base_output_dir, conf_threshold=0.25, save_annotated=False, args=None):
    """
    Evaluate a single model fold
    
    Args:
        model_type (str): Model type (Yolov5, Yolov8, etc.)
        fold (int): Model fold number
        ground_truth_csv (str): Path to ground truth CSV
        images_path (str): Path to test images
        base_output_dir (str): Base output directory
        conf_threshold (float): Confidence threshold
        save_annotated (bool): Whether to save annotated images
        args: Command line arguments
        
    Returns:
        tuple: Metrics dictionary and list of predictions
    """
    # Define model paths
    # Define model paths - replace with actual paths to your model weight files
    model_paths = {
        'Yolov5': {
            1: '<path to the weight model>/Helmet_YOLOv5n/fold_1/weights/best.pt',
            2: '<path to the weight model>/Helmet_YOLOv5n/fold_2/weights/best.pt',
            3: '<path to the weight model>/Helmet_YOLOv5n/fold_3/weights/best.pt',
            4: '<path to the weight model>/Helmet_YOLOv5n/fold_4/weights/best.pt',
            5: '<path to the weight model>/Helmet_YOLOv5n/fold_5/weights/best.pt'
        },
        'Yolov8': {
            1: '<path to the weight model>/Helmet_YOLOv8n/fold_1/weights/best.pt',
            2: '<path to the weight model>/Helmet_YOLOv8n/fold_2/weights/best.pt',
            3: '<path to the weight model>/Helmet_YOLOv8n/fold_3/weights/best.pt',
            4: '<path to the weight model>/Helmet_YOLOv8n/fold_4/weights/best.pt',
            5: '<path to the weight model>/Helmet_YOLOv8n/fold_5/weights/best.pt'
        },
        'Yolov12': {
            1: '<path to the weight model>/Helmet_YOLOv12n/fold_1/weights/best.pt',
            2: '<path to the weight model>/Helmet_YOLOv12n/fold_2/weights/best.pt',
            3: '<path to the weight model>/Helmet_YOLOv12n/fold_3/weights/best.pt',
            4: '<path to the weight model>/Helmet_YOLOv12n/fold_4/weights/best.pt',
            5: '<path to the weight model>/Helmet_YOLOv12n/fold_5/weights/best.pt'
        },
        'RTDETR': {
            1: '<path to the weight model>/Helmet_RTDETR/fold_1/weights/best.pt',
            2: '<path to the weight model>/Helmet_RTDETR/fold_2/weights/best.pt',
            3: '<path to the weight model>/Helmet_RTDETR/fold_3/weights/best.pt',
            4: '<path to the weight model>/Helmet_RTDETR/fold_4/weights/best.pt',
            5: '<path to the weight model>/Helmet_RTDETR/fold_5/weights/best.pt'
        }
    }
    
    # Get model path
    try:
        model_path = model_paths[model_type][fold]
    except KeyError:
        print(f"Invalid model type '{model_type}' or fold number {fold}", flush=True)
        return None, None
    
    # Load ground truth data
    ground_truth = load_ground_truth(ground_truth_csv)
    if not ground_truth:
        print("Failed to load ground truth data", flush=True)
        return None, None
    
    # Process images and calculate metrics
    print(f"Evaluating {model_type} fold {fold}...", flush=True)
    metrics, predictions = process_images(
        images_path, model_path, ground_truth, 
        base_output_dir, model_type, fold, 
        conf_threshold, save_annotated, args
    )
    
    print(f"Fold {fold} metrics:", flush=True)
    if metrics:
        for key, value in metrics.items():
            print(f"{key}: {value}", flush=True)
    
    return metrics, predictions

def evaluate_all_folds(model_type, ground_truth_csv, images_path, base_output_dir, conf_threshold=0.25, save_annotated=False, args=None):
    """
    Evaluate all folds for a specific model type and calculate aggregated metrics
    
    Args:
        model_type (str): Model type (Yolov5, Yolov8, etc.)
        ground_truth_csv (str): Path to ground truth CSV
        images_path (str): Path to test images
        base_output_dir (str): Base output directory
        conf_threshold (float): Confidence threshold
        save_annotated (bool): Whether to save annotated images
        args: Command line arguments
        
    Returns:
        dict: Dictionary with metrics for all folds, mean, and std
    """
    # Initialize results dictionary
    results = {
        'folds': {},
        'mean': {},
        'std': {}
    }
    
    # Process each fold
    for fold in range(1, 6):  # Assuming 5 folds
        metrics, _ = evaluate_model_fold(
            model_type, fold, ground_truth_csv, 
            images_path, base_output_dir, conf_threshold, 
            save_annotated, args
        )
        
        if metrics:
            results['folds'][fold] = metrics
    
    # Calculate mean and std of metrics across folds
    if results['folds']:
        # Get all metric keys
        metric_keys = results['folds'][list(results['folds'].keys())[0]].keys()
        
        # Calculate mean and std for each metric
        for key in metric_keys:
            values = [results['folds'][fold][key] for fold in results['folds'] if key in results['folds'][fold]]
            if values:
                results['mean'][key] = np.mean(values)
                results['std'][key] = np.std(values)
    
    # Save aggregated results to CSV
    model_output_dir = os.path.join(base_output_dir, model_type)
    if results['mean']:
        save_aggregated_results(results, model_type, model_output_dir)
        
        # Generate and save individual model plots
        save_individual_model_plots(results, model_type, model_output_dir)
    
    return results

def save_aggregated_results(results, model_type, output_dir):
    """
    Save aggregated results to CSV
    
    Args:
        results (dict): Results dictionary with metrics
        model_type (str): Model type
        output_dir (str): Output directory
    """
    # Save mean and std to CSV
    csv_path = os.path.join(output_dir, f"{model_type}_aggregated_metrics.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        # Get all metric names
        fieldnames = ['metric', 'mean', 'std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for metric in results['mean'].keys():
            writer.writerow({
                'metric': metric,
                'mean': results['mean'].get(metric, 'N/A'),
                'std': results['std'].get(metric, 'N/A')
            })
    
    print(f"Aggregated metrics saved to {csv_path}", flush=True)

def generate_individual_error_metrics_plot(results, model_type, output_dir):
    """
    Generate error metrics plot for a single model
    
    Args:
        results (dict): Results dictionary with metrics
        model_type (str): Model type
        output_dir (str): Output directory to save the plot
    """
    # Extract metrics
    mse_mean = results['mean']['mse']
    mae_mean = results['mean']['mae']
    rmse_mean = results['mean']['rmse']
    
    mse_std = results['std']['mse']
    mae_std = results['std']['mae']
    rmse_std = results['std']['rmse']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Add padding to the top of the plot
    plt.subplots_adjust(top=0.85)
    
    # Set positions for bars
    x = np.array([0])
    width = 0.25
    
    # Create bars
    plt.bar(x - width, mse_mean, width, label='MSE', color='steelblue', yerr=mse_std, capsize=5)
    plt.bar(x, mae_mean, width, label='MAE', color='darkorange', yerr=mae_std, capsize=5)
    plt.bar(x + width, rmse_mean, width, label='RMSE', color='forestgreen', yerr=rmse_std, capsize=5)
    
    # Add labels and title
    plt.title(f'Error Metrics for {model_type}', pad=20)
    plt.ylabel('Value')
    plt.xticks(x, [model_type])
    plt.legend()
    
    # Add values on top of bars
    plt.text(x - width, mse_mean + mse_std + 0.05, f'{mse_mean:.2f}±{mse_std:.2f}', ha='center')
    plt.text(x, mae_mean + mae_std + 0.05, f'{mae_mean:.2f}±{mae_std:.2f}', ha='center')
    plt.text(x + width, rmse_mean + rmse_std + 0.05, f'{rmse_mean:.2f}±{rmse_std:.2f}', ha='center')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"{model_type}_error_metrics.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Error metrics plot saved to {output_dir}", flush=True)

def generate_individual_classification_metrics_plot(results, model_type, output_dir):
    """
    Generate classification metrics plot for a single model
    
    Args:
        results (dict): Results dictionary with metrics
        model_type (str): Model type
        output_dir (str): Output directory to save the plot
    """
    # Extract metrics
    accuracy_mean = results['mean']['accuracy']
    precision_mean = results['mean']['precision']
    recall_mean = results['mean']['recall']
    f1_mean = results['mean']['f1_score']
    
    accuracy_std = results['std']['accuracy']
    precision_std = results['std']['precision']
    recall_std = results['std']['recall']
    f1_std = results['std']['f1_score']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Add padding to the top of the plot
    plt.subplots_adjust(top=0.85)
    
    # Set positions for bars
    x = np.array([0])
    width = 0.2
    
    # Create bars
    plt.bar(x - 1.5*width, accuracy_mean, width, label='Accuracy', color='steelblue', yerr=accuracy_std, capsize=5)
    plt.bar(x - 0.5*width, precision_mean, width, label='Precision', color='darkorange', yerr=precision_std, capsize=5)
    plt.bar(x + 0.5*width, recall_mean, width, label='Recall', color='forestgreen', yerr=recall_std, capsize=5)
    plt.bar(x + 1.5*width, f1_mean, width, label='F1 Score', color='firebrick', yerr=f1_std, capsize=5)
    
    # Add labels and title
    plt.title(f'Classification Metrics for {model_type}', pad=20)
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.xticks(x, [model_type])
    plt.legend()
    
    # Add values on top of bars
    plt.text(x - 1.5*width, accuracy_mean + accuracy_std + 0.02, f'{accuracy_mean:.2f}±{accuracy_std:.2f}', ha='center')
    plt.text(x - 0.5*width, precision_mean + precision_std + 0.02, f'{precision_mean:.2f}±{precision_std:.2f}', ha='center')
    plt.text(x + 0.5*width, recall_mean + recall_std + 0.02, f'{recall_mean:.2f}±{recall_std:.2f}', ha='center')
    plt.text(x + 1.5*width, f1_mean + f1_std + 0.02, f'{f1_mean:.2f}±{f1_std:.2f}', ha='center')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"{model_type}_classification_metrics.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Classification metrics plot saved to {output_dir}", flush=True)

def save_individual_model_plots(results, model_type, output_dir):
    """
    Generate and save plots for an individual model
    
    Args:
        results (dict): Results dictionary with metrics
        model_type (str): Model type
        output_dir (str): Output directory
    """
    try:
        generate_individual_error_metrics_plot(results, model_type, output_dir)
        generate_individual_classification_metrics_plot(results, model_type, output_dir)
    except Exception as e:
        print(f"Error generating individual plots for {model_type}: {e}", flush=True)

def generate_comparative_classification_plot(all_results, output_dir):
    """
    Generate a comparative classification metrics plot for all models
    
    Args:
        all_results (dict): Dictionary with results for all models
        output_dir (str): Output directory to save the plot
    """
    # Get model types with valid metrics
    model_types = [model for model in all_results.keys() 
                  if 'mean' in all_results[model] 
                  and 'accuracy' in all_results[model]['mean']]
    
    num_models = len(model_types)
    
    if num_models == 0:
        print("No models with valid metrics to compare", flush=True)
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Add padding to the top of the plot
    plt.subplots_adjust(top=0.85)
    
    # Set positions for bars
    x = np.arange(num_models)
    width = 0.2
    
    # Extract metrics for all models
    accuracy_values = [all_results[model]['mean']['accuracy'] for model in model_types]
    precision_values = [all_results[model]['mean']['precision'] for model in model_types]
    recall_values = [all_results[model]['mean']['recall'] for model in model_types]
    f1_values = [all_results[model]['mean']['f1_score'] for model in model_types]
    
    accuracy_std = [all_results[model]['std']['accuracy'] for model in model_types]
    precision_std = [all_results[model]['std']['precision'] for model in model_types]
    recall_std = [all_results[model]['std']['recall'] for model in model_types]
    f1_std = [all_results[model]['std']['f1_score'] for model in model_types]
    
    # Create grouped bars
    plt.bar(x - 1.5*width, accuracy_values, width, label='Accuracy', color='steelblue', yerr=accuracy_std, capsize=5)
    plt.bar(x - 0.5*width, precision_values, width, label='Precision', color='darkorange', yerr=precision_std, capsize=5)
    plt.bar(x + 0.5*width, recall_values, width, label='Recall', color='forestgreen', yerr=recall_std, capsize=5)
    plt.bar(x + 1.5*width, f1_values, width, label='F1 Score', color='firebrick', yerr=f1_std, capsize=5)
    
    # Add labels and title
    plt.title('Classification Metrics Comparison Across Models', pad=20)
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.xticks(x, model_types)
    plt.legend()
    
    # Add values on top of bars
    for i in range(num_models):
        plt.text(i - 1.5*width, accuracy_values[i] + accuracy_std[i] + 0.02, f'{accuracy_values[i]:.2f}', ha='center', fontsize=8)
        plt.text(i - 0.5*width, precision_values[i] + precision_std[i] + 0.02, f'{precision_values[i]:.2f}', ha='center', fontsize=8)
        plt.text(i + 0.5*width, recall_values[i] + recall_std[i] + 0.02, f'{recall_values[i]:.2f}', ha='center', fontsize=8)
        plt.text(i + 1.5*width, f1_values[i] + f1_std[i] + 0.02, f'{f1_values[i]:.2f}', ha='center', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_classification_metrics.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Comparative classification metrics plot saved to {output_dir}", flush=True)

def generate_comparative_error_plot(all_results, output_dir):
    """
    Generate a comparative error metrics plot for all models
    
    Args:
        all_results (dict): Dictionary with results for all models
        output_dir (str): Output directory to save the plot
    """
    # Get model types with valid metrics
    model_types = [model for model in all_results.keys() 
                  if 'mean' in all_results[model] 
                  and 'mse' in all_results[model]['mean']]
    
    num_models = len(model_types)
    
    if num_models == 0:
        print("No models with valid metrics to compare", flush=True)
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Add padding to the top of the plot
    plt.subplots_adjust(top=0.85)
    
    # Set positions for bars
    x = np.arange(num_models)
    width = 0.25
    
    # Extract metrics for all models
    mse_values = [all_results[model]['mean']['mse'] for model in model_types]
    mae_values = [all_results[model]['mean']['mae'] for model in model_types]
    rmse_values = [all_results[model]['mean']['rmse'] for model in model_types]
    
    mse_std = [all_results[model]['std']['mse'] for model in model_types]
    mae_std = [all_results[model]['std']['mae'] for model in model_types]
    rmse_std = [all_results[model]['std']['rmse'] for model in model_types]
    
    # Create grouped bars
    plt.bar(x - width, mse_values, width, label='MSE', color='steelblue', yerr=mse_std, capsize=5)
    plt.bar(x, mae_values, width, label='MAE', color='darkorange', yerr=mae_std, capsize=5)
    plt.bar(x + width, rmse_values, width, label='RMSE', color='forestgreen', yerr=rmse_std, capsize=5)
    
    # Add labels and title
    plt.title('Error Metrics Comparison Across Models', pad=20)
    plt.ylabel('Value')
    plt.xticks(x, model_types)
    plt.legend()
    
    # Add values on top of bars
    for i in range(num_models):
        plt.text(i - width, mse_values[i] + mse_std[i] + 0.05, f'{mse_values[i]:.2f}', ha='center', fontsize=8)
        plt.text(i, mae_values[i] + mae_std[i] + 0.05, f'{mae_values[i]:.2f}', ha='center', fontsize=8)
        plt.text(i + width, rmse_values[i] + rmse_std[i] + 0.05, f'{rmse_values[i]:.2f}', ha='center', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_error_metrics.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Comparative error metrics plot saved to {output_dir}", flush=True)

def generate_summary_report(all_results, conf_threshold, output_dir):
    """
    Generate a summary report of all model performances
    
    Args:
        all_results (dict): Dictionary with results for all models
        conf_threshold (float): Confidence threshold used
        output_dir (str): Output directory to save the report
    """
    # Get model types
    model_types = list(all_results.keys())
    
    if not model_types:
        print("No results to summarize", flush=True)
        return
    
    # Create summary report
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"HELMET DETECTION MODELS EVALUATION SUMMARY\n")
        f.write(f"Confidence Threshold: {conf_threshold}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Classification metrics summary
        f.write("-" * 80 + "\n")
        f.write("CLASSIFICATION METRICS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<10} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}\n")
        
        for model in model_types:
            acc = f"{all_results[model]['mean']['accuracy']:.4f}±{all_results[model]['std']['accuracy']:.4f}"
            prec = f"{all_results[model]['mean']['precision']:.4f}±{all_results[model]['std']['precision']:.4f}"
            rec = f"{all_results[model]['mean']['recall']:.4f}±{all_results[model]['std']['recall']:.4f}"
            f1 = f"{all_results[model]['mean']['f1_score']:.4f}±{all_results[model]['std']['f1_score']:.4f}"
            
            f.write(f"{model:<10} {acc:<15} {prec:<15} {rec:<15} {f1:<15}\n")
        
        # Error metrics summary
        f.write("\n" + "-" * 80 + "\n")
        f.write("ERROR METRICS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<10} {'MSE':<15} {'MAE':<15} {'RMSE':<15}\n")
        
        for model in model_types:
            mse = f"{all_results[model]['mean']['mse']:.4f}±{all_results[model]['std']['mse']:.4f}"
            mae = f"{all_results[model]['mean']['mae']:.4f}±{all_results[model]['std']['mae']:.4f}"
            rmse = f"{all_results[model]['mean']['rmse']:.4f}±{all_results[model]['std']['rmse']:.4f}"
            
            f.write(f"{model:<10} {mse:<15} {mae:<15} {rmse:<15}\n")
        
        # Best model identification
        f.write("\n" + "-" * 80 + "\n")
        f.write("BEST MODEL ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Find best model for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'mse', 'mae', 'rmse']
        best_models = {}
        
        for metric in metrics:
            if metric in ['mse', 'mae', 'rmse']:  # Lower is better
                best_value = float('inf')
                best_model = None
                for model in model_types:
                    if all_results[model]['mean'][metric] < best_value:
                        best_value = all_results[model]['mean'][metric]
                        best_model = model
            else:  # Higher is better
                best_value = -float('inf')
                best_model = None
                for model in model_types:
                    if all_results[model]['mean'][metric] > best_value:
                        best_value = all_results[model]['mean'][metric]
                        best_model = model
            
            best_models[metric] = (best_model, best_value)
        
        # Write best models
        for metric, (model, value) in best_models.items():
            f.write(f"Best model for {metric}: {model} ({value:.4f})\n")
        
        # Additional analysis and recommendations
        f.write("\n" + "-" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Count how many times each model is the best
        model_counts = {}
        for model in model_types:
            model_counts[model] = 0
        
        for metric, (model, _) in best_models.items():
            model_counts[model] += 1
        
        # Find the overall best model(s)
        max_count = max(model_counts.values())
        best_overall = [model for model, count in model_counts.items() if count == max_count]
        
        f.write(f"Overall best model(s) based on performance across metrics: {', '.join(best_overall)}\n\n")
        
        # Add specific recommendations
        f.write("Specific recommendations:\n")
        
        # For classification performance
        best_f1_model, _ = best_models['f1_score']
        f.write(f"- For best classification performance (helmet presence): {best_f1_model}\n")
        
        # For count accuracy
        best_mae_model, _ = best_models['mae']
        f.write(f"- For best helmet count accuracy: {best_mae_model}\n")
        
        # Overall recommendation
        f.write(f"- Overall recommendation: {best_overall[0]} provides the best balance of metrics\n")
        
        # Final notes
        f.write("\n" + "-" * 80 + "\n")
        f.write("NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("- Results are averaged across 5 cross-validation folds\n")
        f.write(f"- Confidence threshold: {conf_threshold}\n")
        f.write("- Values are reported as mean±std\n")
        f.write("- Lower values are better for MSE, MAE, and RMSE\n")
        f.write("- Higher values are better for Accuracy, Precision, Recall, and F1 Score\n")
    
    print(f"Summary report saved to {report_path}", flush=True)
    
    # Also create a CSV version for easier analysis
    csv_path = os.path.join(output_dir, "summary_metrics.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'accuracy', 'accuracy_std', 'precision', 'precision_std',
                      'recall', 'recall_std', 'f1_score', 'f1_score_std',
                      'mse', 'mse_std', 'mae', 'mae_std', 'rmse', 'rmse_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for model in model_types:
            writer.writerow({
                'model': model,
                'accuracy': all_results[model]['mean']['accuracy'],
                'accuracy_std': all_results[model]['std']['accuracy'],
                'precision': all_results[model]['mean']['precision'],
                'precision_std': all_results[model]['std']['precision'],
                'recall': all_results[model]['mean']['recall'],
                'recall_std': all_results[model]['std']['recall'],
                'f1_score': all_results[model]['mean']['f1_score'],
                'f1_score_std': all_results[model]['std']['f1_score'],
                'mse': all_results[model]['mean']['mse'],
                'mse_std': all_results[model]['std']['mse'],
                'mae': all_results[model]['mean']['mae'],
                'mae_std': all_results[model]['std']['mae'],
                'rmse': all_results[model]['mean']['rmse'],
                'rmse_std': all_results[model]['std']['rmse']
            })
    
    print(f"Summary metrics CSV saved to {csv_path}", flush=True)

def evaluate_all_models(model_types, ground_truth_csv, images_path, conf_threshold=0.25, save_annotated=False, args=None):
    """
    Evaluate all specified models and generate comparative results
    
    Args:
        model_types (list): List of model types to evaluate
        ground_truth_csv (str): Path to ground truth CSV
        images_path (str): Path to test images
        conf_threshold (float): Confidence threshold
        save_annotated (bool): Whether to save annotated images
        args: Command line arguments
        
    Returns:
        dict: Dictionary with results for all models
    """
    # Create base output directory
    base_output_dir = create_output_directory(args)
    
    # Initialize results dictionary
    all_results = {}
    
    # Evaluate each model type
    for model_type in model_types:
        print(f"\n{'='*50}\nEvaluating {model_type}\n{'='*50}", flush=True)
        
        # Set correct model type in args to ensure proper loading
        args.model_type = model_type
        
        # Evaluate all folds for this model type
        results = evaluate_all_folds(
            model_type, ground_truth_csv, 
            images_path, base_output_dir, conf_threshold, 
            save_annotated, args
        )
        
        # Store results only if there are valid metrics
        if results and 'mean' in results and results['mean']:
            all_results[model_type] = results
    
    # Generate comparative plots only if we have results
    if all_results:
        summary_dir = os.path.join(base_output_dir, "Summary")
        
        # Check if any model has valid metrics before generating plots
        if any('accuracy' in all_results[model]['mean'] for model in all_results):
            generate_comparative_classification_plot(all_results, summary_dir)
            generate_comparative_error_plot(all_results, summary_dir)
            generate_summary_report(all_results, conf_threshold, summary_dir)
        else:
            print("No valid metrics found for any model. Skipping comparative plots.", flush=True)
            
            # Create a basic error report
            with open(os.path.join(summary_dir, "error_report.txt"), 'w') as f:
                f.write("ERROR: No valid metrics were generated for any model.\n\n")
                f.write("Possible causes:\n")
                f.write("1. Could not load model files correctly\n")
                f.write("2. Error in setting class names for models\n")
                f.write("3. No valid predictions were generated\n\n")
                f.write(f"Check individual model folders for more details.\n")
    else:
        print("No results were generated for any model.", flush=True)
    
    print(f"\nAll model evaluations complete. Results saved to {base_output_dir}", flush=True)
    
    return all_results, base_output_dir

def main():
    """
    Main function to run the evaluation with command-line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate helmet detection models')
    parser.add_argument('--model-type', type=str, nargs='+',
                        default=['Yolov5', 'Yolov8', 'Yolov12', 'RTDETR'],
                        help='Model type(s) to evaluate')
    parser.add_argument('--ground-truth', type=str, required=True,
                        help='Path to ground truth CSV file')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--save-annotated', action='store_true',
                        help='Save annotated images')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to evaluate (if not specified, evaluates all folds)')
    parser.add_argument('--redtr', action='store_true',
                        help='Use RT-DETR model type')
    parser.add_argument('--dir_name', type=str, default=None,
                        help='Directory name for output')

    args = parser.parse_args()
    
    print("#" * 50, flush=True)
    print(f"Starting evaluation for model(s): {', '.join(args.model_type)}", flush=True)
    print(f"Ground truth: {args.ground_truth}", flush=True)
    print(f"Test images: {args.images}", flush=True)
    print(f"Confidence threshold: {args.conf}", flush=True)
    print(f"Save annotated images: {args.save_annotated}", flush=True)
    print("#" * 50, flush=True)

    
    if args.fold is not None:
        # Evaluate specific fold for a single model
        if len(args.model_type) != 1:
            print("When specifying a fold, please provide only one model type", flush=True)
            return
        
        model_type = args.model_type[0]
        base_output_dir = create_output_directory(args)
        
        metrics, _, _ = evaluate_model_fold(
            model_type, args.fold, args.ground_truth, 
            args.images, base_output_dir, args.conf, 
            args.save_annotated, args
        )
    else:
        # Evaluate all models
        evaluate_all_models(
            args.model_type, args.ground_truth,
            args.images, args.conf,
            args.save_annotated, args
        )
    
    print("\nEvaluation completed.", flush=True)

if __name__ == "__main__":
    main()