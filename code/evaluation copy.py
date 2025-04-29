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

def load_model(model_path, args):
    """
    Load the YOLO model from the specified path and set proper class names
    """
    try:
        print(f"Loading model from {model_path}...", flush=True)
        if args.redtr:
            model = RTDETR(model_path)
        else:
            model = YOLO(model_path)
        print(f"Model loaded successfully", flush=True)
        return model
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return None

def predict_helmet(image_path, model, conf_threshold=0.25):
    """
    Predict helmets in an image using the loaded model
    
    Args:
        image_path (str): Path to the image
        model: Loaded YOLO model
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        dict: Contains information about helmet presence and count
    """
    print(f"Processing image: {image_path}", flush=True)
    
    # Initialize result dictionary
    prediction_result = {
        "image_name": os.path.basename(image_path),
        "helmet_presence": 0,
        "helmet_count": 0
    }
    
    try:
        # Run prediction
        results = model(image_path, conf=conf_threshold, verbose=False)[0]
        
        # Filter for only helmet class (class 0)
        helmet_detections = [det for det in results.boxes.data.tolist() 
                             if int(det[5]) == 0 and det[4] >= conf_threshold]
        
        # Update result with helmet information
        helmet_count = len(helmet_detections)
        prediction_result["helmet_count"] = helmet_count
        prediction_result["helmet_presence"] = 1 if helmet_count > 0 else 0
        
        return prediction_result, results
    except Exception as e:
        print(f"Error during prediction: {e}", flush=True)
        return prediction_result, None

def create_output_directory(base_folder_name, args):
    """
    Create output directory structure with timestamp
    
    Args:
        base_folder_name (str): Base name for the output folder
        
    Returns:
        str: Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Directory_name = args.dir_name if args.dir_name else "Final_Output"
    output_dir = os.path.join(Directory_name, f"{base_folder_name}_{args.conf}_{timestamp}")
    
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output directory: {output_dir}", flush=True)
    return output_dir

def save_annotated_image(results, output_dir, image_name, helmet_class=0):
    """
    Save annotated image with bounding boxes for helmets only
    
    Args:
        results: YOLO prediction results
        output_dir (str): Directory to save the image
        image_name (str): Original image name
        helmet_class (int): Class ID for helmets
    """
    # Get the original image
    orig_img = results.orig_img.copy()
    
    # Filter detections for helmet class only
    helmet_detections = []
    for det in results.boxes.data.tolist():
        if int(det[5]) == helmet_class:
            x1, y1, x2, y2, conf, cls = det[:6]
            helmet_detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    
    # If no helmet detections, skip saving this image
    if not helmet_detections:
        return
    
    # Draw bounding boxes ONLY for helmet detections
    for det in helmet_detections:
        x1, y1, x2, y2, conf = det
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, f"Helmet: {conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the image only if we drew helmet boxes
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, orig_img)
    print(f"Saved annotated image with {len(helmet_detections)} helmet detections to {output_path}")

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

def process_images(image_paths, model_path, ground_truth, output_dir, fold_num, conf_threshold=0.25, save_annotated=False, args=None):
    """
    Process images and calculate metrics for a single model fold
    
    Args:
        image_paths (str or list): Directory containing images or list of image paths
        model_path (str): Path to the YOLO model
        ground_truth (dict): Ground truth data
        output_dir (str): Output directory for results
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
        print("Failed to load model, skipping this fold", flush=True)
        return None, None
    
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
    for img_path in tqdm(image_paths, desc="Processing images", unit="image"):
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
    csv_path = os.path.join(output_dir, f"{args.model_type}_fold_{fold_num}_predictions.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'helmet_presence', 'helmet_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for pred in predictions:
            writer.writerow(pred)
    
    print(f"Predictions for fold {fold_num} saved to {csv_path}", flush=True)
    
    return metrics, predictions

def evaluate_model_fold(model_type, fold, ground_truth_csv, images_path, conf_threshold=0.25, save_annotated=False, args=None):
    """
    Evaluate a single model fold
    
    Args:
        model_type (str): Model type (Yolov5, Yolov8, etc.)
        fold (int): Model fold number
        ground_truth_csv (str): Path to ground truth CSV
        images_path (str): Path to test images
        conf_threshold (float): Confidence threshold
        save_annotated (bool): Whether to save annotated images
        args: Command line arguments
        
    Returns:
        tuple: Metrics dictionary and list of predictions
    """
    # Define model paths
    model_paths = {
        'Yolov5': {
            1: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv5n/fold_1/weights/best.pt',
            2: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv5n/fold_2/weights/best.pt',
            3: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv5n/fold_3/weights/best.pt',
            4: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv5n/fold_4/weights/best.pt',
            5: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv5n/fold_5/weights/best.pt'
        },
        'Yolov8': {
            1: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv8n/fold_13/weights/best.pt',
            2: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv8n/fold_2/weights/best.pt',
            3: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv8n/fold_3/weights/best.pt',
            4: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv8n/fold_4/weights/best.pt',
            5: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv8n/fold_5/weights/best.pt'
        },
        'Yolov12': {
            1: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv12n/fold_1/weights/best.pt',
            2: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv12n/fold_2/weights/best.pt',
            3: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv12n/fold_3/weights/best.pt',
            4: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv12n/fold_4/weights/best.pt',
            5: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv12n/fold_5/weights/best.pt'
        },
        'RTDETR': {
            1: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_RTDETR/fold_1/weights/best.pt',
            2: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_RTDETR/fold_2/weights/best.pt',
            3: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_RTDETR/fold_3/weights/best.pt',
            4: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_RTDETR/fold_4/weights/best.pt',
            5: '/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_RTDETR/fold_5/weights/best.pt'
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
    
    # Create output directory based on model type
    output_dir = create_output_directory(f"{model_type}", args=args)
    
    # Process images and calculate metrics
    print(f"Evaluating {model_type} fold {fold}...", flush=True)
    metrics, predictions = process_images(
        images_path, model_path, ground_truth, 
        output_dir, fold, conf_threshold, 
        save_annotated, args
    )
    
    print(f"Fold {fold} metrics:", flush=True)
    if metrics:
        for key, value in metrics.items():
            print(f"{key}: {value}", flush=True)
    
    return metrics, predictions, output_dir

def evaluate_all_folds(model_type, ground_truth_csv, images_path, conf_threshold=0.25, save_annotated=False, args=None):
    """
    Evaluate all folds for a specific model type and calculate aggregated metrics
    
    Args:
        model_type (str): Model type (Yolov5, Yolov8, etc.)
        ground_truth_csv (str): Path to ground truth CSV
        images_path (str): Path to test images
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
    
    output_dir = None
    
    # Process each fold
    for fold in range(1, 6):  # Assuming 5 folds
        metrics, _, fold_output_dir = evaluate_model_fold(
            model_type, fold, ground_truth_csv, 
            images_path, conf_threshold, 
            save_annotated, args
        )
        
        if metrics:
            results['folds'][fold] = metrics
            if output_dir is None:
                output_dir = fold_output_dir
    
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
    if output_dir and results['mean']:
        save_aggregated_results(results, model_type, output_dir)
        
        # Generate and save plots
        save_plots(results, model_type, output_dir)
    
    return results, output_dir

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

def generate_error_metrics_plot(results, model_type, output_dir):
    """
    Generate error metrics comparison plot
    
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
    plt.figure(figsize=(10, 6))
    
    # Set positions for bars
    x = np.array([0])
    width = 0.25
    
    # Create bars
    plt.bar(x - width, mse_mean, width, label='MSE', color='steelblue', yerr=mse_std, capsize=5)
    plt.bar(x, mae_mean, width, label='MAE', color='darkorange', yerr=mae_std, capsize=5)
    plt.bar(x + width, rmse_mean, width, label='RMSE', color='forestgreen', yerr=rmse_std, capsize=5)
    
    # Add labels and title
    plt.title(f'Error Metrics for {model_type}')
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

def generate_classification_metrics_plot(results, model_type, output_dir):
    """
    Generate classification metrics comparison plot
    
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
    plt.figure(figsize=(10, 6))
    
    # Set positions for bars
    x = np.array([0])
    width = 0.2
    
    # Create bars
    plt.bar(x - 1.5*width, accuracy_mean, width, label='Accuracy', color='steelblue', yerr=accuracy_std, capsize=5)
    plt.bar(x - 0.5*width, precision_mean, width, label='Precision', color='darkorange', yerr=precision_std, capsize=5)
    plt.bar(x + 0.5*width, recall_mean, width, label='Recall', color='forestgreen', yerr=recall_std, capsize=5)
    plt.bar(x + 1.5*width, f1_mean, width, label='F1 Score', color='firebrick', yerr=f1_std, capsize=5)
    
    # Add labels and title
    plt.title(f'Classification Metrics for {model_type}')
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

def save_plots(results, model_type, output_dir):
    """
    Generate and save all plots
    
    Args:
        results (dict): Results dictionary with metrics
        model_type (str): Model type
        output_dir (str): Output directory
    """
    try:
        generate_error_metrics_plot(results, model_type, output_dir)
        generate_classification_metrics_plot(results, model_type, output_dir)
    except Exception as e:
        print(f"Error generating plots: {e}", flush=True)

def main():
    """
    Main function to run the evaluation with command-line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate helmet detection models')
    parser.add_argument('--model-type', type=str, required=True, 
                        choices=['Yolov5', 'Yolov8', 'Yolov12', 'RTDETR'],
                        help='Model type (Yolov5, Yolov8, Yolov12, RTDETR)')
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
    parser.add_argument('--dir_name', type=str,default=None,
                        help='Directory name for output')

    args = parser.parse_args()
    
    print("#" * 50, flush=True)
    print(f"Starting evaluation for {args.model_type}", flush=True)
    print(f"Ground truth: {args.ground_truth}", flush=True)
    print(f"Test images: {args.images}", flush=True)
    print(f"Confidence threshold: {args.conf}", flush=True)
    print(f"Save annotated images: {args.save_annotated}", flush=True)
    if args.fold is not None:
        print(f"Evaluating only fold {args.fold}", flush=True)
    else:
        print(f"Evaluating all folds", flush=True)
    print("#" * 50, flush=True)
    
    if args.fold is not None:
        # Evaluate specific fold
        metrics, _, _ = evaluate_model_fold(
            args.model_type, args.fold, args.ground_truth, 
            args.images, args.conf, args.save_annotated, args
        )
    else:
        # Evaluate all folds
        results, _ = evaluate_all_folds(
            args.model_type, args.ground_truth, 
            args.images, args.conf, args.save_annotated, args
        )
        
        if results['mean']:
            print("\nAggregated Metrics (Mean ± Std):", flush=True)
            for key in results['mean']:
                print(f"{key}: {results['mean'][key]:.4f} ± {results['std'][key]:.4f}", flush=True)
    
    print("\nEvaluation completed.", flush=True)

if __name__ == "__main__":
    # Import additional requirements inside __main__ to reduce import time for modules
    import os
    import csv
    import cv2
    from datetime import datetime
    from pathlib import Path
    import argparse
    
    main()