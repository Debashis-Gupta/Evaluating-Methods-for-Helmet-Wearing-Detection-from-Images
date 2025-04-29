import os
import csv
from datetime import datetime
import torch
from ultralytics import YOLO, RTDETR
from pathlib import Path
import cv2
import random
random.seed(42)

def load_model(model_path, args):
    """
    Load the YOLO or RTDETR model from the specified path and set proper class names
    """
    try:
        if args.redtr:
            print("Loading RTDETR model...", flush=True)
            model = RTDETR(model_path)
            
            # RTDETR-specific class name handling
            if hasattr(model, 'model') and hasattr(model.model, 'class_names'):
                model.model.class_names = ['helmet', 'driver', 'passenger']
                print("Set RTDETR class names using model.model.class_names", flush=True)
        else:
            print("Loading YOLO model...", flush=True)
            model = YOLO(model_path)
            
            # YOLO-specific class name handling
            if hasattr(model, 'names'):
                model.names = {0: 'helmet', 1: 'driver', 2: 'passenger'}
                print("Set YOLO class names using model.names", flush=True)
            elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                model.model.names = {0: 'helmet', 1: 'driver', 2: 'passenger'}
                print("Set YOLO class names using model.model.names", flush=True)
        
        print(f"Model successfully loaded from {model_path}", flush=True)
        
        # Display final class names for verification
        if hasattr(model, 'names'):
            print(f"Final model class names: {model.names}", flush=True)
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'names'):
                print(f"Final model class names: {model.model.names}", flush=True)
            elif hasattr(model.model, 'class_names'):
                print(f"Final model class names: {model.model.class_names}", flush=True)
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def predict_helmet(image_path, model, conf_threshold=0.25):
    """
    Predict helmets in an image using the loaded model
    
    Args:
        image_path (str): Path to the image
        model: Loaded YOLO or RTDETR model
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
        # Check if model is RTDETR or YOLO
        is_rtdetr = isinstance(model, RTDETR)
        
        # Run prediction with model-specific parameters if needed
        if is_rtdetr:
            # RTDETR specific parameters if needed
            print(f"Running prediction with RTDETR model", flush=True)
            results = model(image_path, conf=conf_threshold)[0]
            
            # For RTDETR, print specific model information
            if hasattr(model, 'model') and hasattr(model.model, 'class_names'):
                print(f"RTDETR class names: {model.model.class_names}", flush=True)
        else:
            # Standard YOLO parameters
            print(f"Running prediction with YOLO model", flush=True)
            results = model(image_path, conf=conf_threshold)[0]
            
            # For YOLO, print specific model information
            if hasattr(model, 'names'):
                print(f"YOLO class names: {model.names}", flush=True)
        
        # Get helmet class ID (always 0 for our models)
        helmet_class = 0
        
        # Process detections
        helmet_detections = []
        for det in results.boxes.data.tolist():
            class_id = int(det[5])
            confidence = float(det[4])
            
            if class_id == helmet_class and confidence >= conf_threshold:
                helmet_detections.append(det)
        
        # Update result dictionary
        if len(helmet_detections) > 0:
            prediction_result["helmet_presence"] = 1
            prediction_result["helmet_count"] = len(helmet_detections)
        
        return prediction_result, results
    except Exception as e:
        print(f"Error during prediction: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return prediction_result, None

def create_output_directory(base_folder_name,args):
    """
    Create output directory structure with timestamp
    
    Args:
        base_folder_name (str): Base name for the output folder
        
    Returns:
        str: Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("full_predict", f"{base_folder_name}_{args.conf}_{timestamp}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output directory: {output_dir}",flush=True)
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
        cls = int(det[5])
        if cls == helmet_class:
            helmet_detections.append(det)
    
    # If no helmet detections, skip saving this image
    if not helmet_detections:
        print(f"No helmet detections in {image_name}, skipping annotation")
        return
    
    # Draw bounding boxes ONLY for helmet detections
    for det in helmet_detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        
        
        # Draw text (black text on green background)
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
    print(f"Saved annotated image with {len(helmet_detections)} helmet detections to {output_path}")

def process_images(image_paths, model_path, output_folder_name, conf_threshold=0.25, save_annotated=True,args=None):
    """
    Process multiple images and generate CSV report
    
    Args:
        image_paths (str or list): Directory containing images or list of image paths
        model_path (str): Path to the YOLO model
        output_folder_name (str): Base name for output folder
        conf_threshold (float): Confidence threshold for detections
        save_annotated (bool): Whether to save annotated images
    """
    # Load model
    model = load_model(model_path,args=args)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Create output directory
    output_dir = create_output_directory(output_folder_name,args=args)
    
    # Initialize results list for CSV
    results_list = []
    
    # Handle directory or list of image paths
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        print(f"Processing images from directory: {image_paths}",flush=True)
        # Get all image files from directory
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        for file in os.listdir(image_paths):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_files.append(os.path.join(image_paths, file))
        if not image_files:
            print(f"No valid image files found in {image_paths}",flush=True)
            return
        image_paths = image_files
    
    # Process each image
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.",flush=True)
            continue
        
        # Get image name
        img_name = os.path.basename(img_path)
        
        # Process image
        prediction, yolo_results = predict_helmet(img_path, model, conf_threshold)
        results_list.append(prediction)
        
        # Save annotated image if requested
        if save_annotated:
            save_annotated_image(yolo_results, output_dir, img_name)
    
    # Create CSV file
    csv_path = os.path.join(output_dir, "helmet_detection_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'helmet_presence', 'helmet_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results_list:
            writer.writerow(result)
    
    print(f"Results saved to {csv_path}",flush=True)
    return results_list

def main():
    """
    Main function to run the prediction with command-line arguments
    """
    import argparse
    model="/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/code/Helmet_YOLOv5n/fold_1/weights/best.pt"
    images_path="/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/test_combined_output/images"
    output="YOLO8n"
    parser = argparse.ArgumentParser(description='Detect helmets in images using YOLO')
    parser.add_argument('--model', type=str, default=model, help='Path to the YOLO model')
    parser.add_argument('--csv_name', type=str, help='Provide CSV Name')
    parser.add_argument('--images', type=str, nargs='+', default=images_path, help='Path(s) to image file(s)')
    parser.add_argument('--output', type=str, default=output, help='Output folder name')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save-annotated', action='store_true', help='Save annotated images')
    parser.add_argument('--redtr', action='store_true', help='REDTR mode')
    
    args = parser.parse_args()
    print("#"*20,flush=True)
    print(f"Confidence threshold: {args.conf}",flush=True)
    print(f"Output folder name: {args.output}",flush=True)
    print(f"Model path: {args.model}",flush=True)
    print(f"Images path: {args.images}",flush=True)
    print(f"Save annotated images: {args.save_annotated}",flush=True)
    print(f"CSV name: {args.csv_name}",flush=True)
    print("#"*20,flush=True)

    model_paths = {
            'Yolov5': {
                1: 'path/to/Helmet_YOLOv5n/fold_1/weights/best.pt',
                2: 'path/to/Helmet_YOLOv5n/fold_2/weights/best.pt',
                3: 'path/to/Helmet_YOLOv5n/fold_3/weights/best.pt',
                4: 'path/to/Helmet_YOLOv5n/fold_4/weights/best.pt',
                5: 'path/to/Helmet_YOLOv5n/fold_5/weights/best.pt'
            },
            'Yolov8': {
                1: 'path/to/Helmet_YOLOv8n/fold_1/weights/best.pt',
                2: 'path/to/Helmet_YOLOv8n/fold_2/weights/best.pt',
                3: 'path/to/Helmet_YOLOv8n/fold_3/weights/best.pt',
                4: 'path/to/Helmet_YOLOv8n/fold_4/weights/best.pt',
                5: 'path/to/Helmet_YOLOv8n/fold_5/weights/best.pt'
            },
            'Yolov12': {
                1: 'path/to/Helmet_YOLOv12n/fold_1/weights/best.pt',
                2: 'path/to/Helmet_YOLOv12n/fold_2/weights/best.pt',
                3: 'path/to/Helmet_YOLOv12n/fold_3/weights/best.pt',
                4: 'path/to/Helmet_YOLOv12n/fold_4/weights/best.pt',
                5: 'path/to/Helmet_YOLOv12n/fold_5/weights/best.pt'
            },
            'RTDETR': {
                1: 'path/to/Helmet_RTDETR/fold_1/weights/best.pt',
                2: 'path/to/Helmet_RTDETR/fold_2/weights/best.pt',
                3: 'path/to/Helmet_RTDETR/fold_3/weights/best.pt',
                4: 'path/to/Helmet_RTDETR/fold_4/weights/best.pt',
                5: 'path/to/Helmet_RTDETR/fold_5/weights/best.pt'
            }
        }
    process_images(args.images, args.model, args.output, args.conf, args.save_annotated,args=args)

if __name__ == "__main__":
    main()