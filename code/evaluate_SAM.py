#!/usr/bin/env python3

import sys
import argparse
import os
import glob
import logging
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import clip
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error
)
import matplotlib.pyplot as plt


def parse_arguments():
    """
    Parse and return command-line arguments with detailed help descriptions.

    Returns:
        argparse.Namespace: Parsed argument namespace containing configuration parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "One-shot helmet counter pipeline: load SAM, generate masks, "
            "compute CLIP similarities, and evaluate metrics over thresholds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--sam-weights",
        type=str,
        default="sam_vit_h.pth",
        help=(
            "Path to the pretrained SAM checkpoint (.pth file). "
            "This checkpoint will be loaded to initialize the SAM model for mask generation."
        )
    )
    parser.add_argument(
        "--yolo-img-dir",
        type=str,
        default="images",
        help="Directory containing YOLO detection images corresponding to label files."
    )
    parser.add_argument(
        "--yolo-lbl-dir",
        type=str,
        default="labels",
        help="Directory containing YOLO label files (.txt) with bounding box annotations."
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="target_images",
        help="Directory containing the target images to be evaluated."
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        default="ground_truth.csv",
        help=(
            "Path to the ground truth CSV file. Columns should include: "
            "image_name, helmet_presence (0/1), helmet_count (integer)."
        )
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="Minimum mask pixel area to consider for candidate helmet segments."
    )
    parser.add_argument(
        "--thresh-start",
        type=float,
        default=0.0,
        help="Starting threshold value for similarity sweep."
    )
    parser.add_argument(
        "--thresh-end",
        type=float,
        default=1.0,
        help="Ending threshold value for similarity sweep."
    )
    parser.add_argument(
        "--thresh-step",
        type=float,
        default=0.01,
        help="Increment step for threshold sweep iterations."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="metrics_by_threshold.csv",
        help="Output path for saving computed metrics as a CSV file."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=(
            "Compute device selection: 'auto' picks GPU if available, "
            "otherwise CPU. Can be manually overridden to 'cuda' or 'cpu'."
        )
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="If set, save metric plots to disk instead of displaying interactively."
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default="metrics_plot.png",
        help="Filename for saving the metrics plot image when --save-plots is used."
    )

    return parser.parse_args()


def main():
    """
    Execute the helmet counting workflow end-to-end with progress monitoring.
    """
    # Configure logging for debug and info messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    args = parse_arguments()
    logging.info("Parsed command-line arguments successfully.")

    # Determine the compute device based on availability and user preference
    if args.device == "auto":
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        compute_device = args.device
    logging.info(f"Using compute device: {compute_device}")

    # STEP 1: Locate a reference helmet bounding box from YOLO label files
    reference_bounding_box = None
    reference_image_path = None
    logging.info("Searching for first helmet (class 0) box in YOLO labels...")

    for label_file in sorted(glob.glob(os.path.join(args.yolo_lbl_dir, "*.txt"))):
        with open(label_file, 'r') as lf:
            for line in lf:
                tokens = line.strip().split()
                if not tokens:
                    continue
                class_id = int(tokens[0])
                if class_id == 0:
                    x_center, y_center, box_width, box_height = map(float, tokens[1:])
                    base_name = os.path.splitext(os.path.basename(label_file))[0]
                    for extension in [".jpg", ".png", ".jpeg"]:
                        candidate = os.path.join(args.yolo_img_dir, base_name + extension)
                        if os.path.exists(candidate):
                            reference_image_path = candidate
                            break
                    if not reference_image_path:
                        raise FileNotFoundError(
                            f"Image file for label {label_file} not found in {args.yolo_img_dir}."
                        )

                    img_arr = cv2.imread(reference_image_path)
                    img_h, img_w = img_arr.shape[:2]
                    x1 = int((x_center - box_width / 2) * img_w)
                    y1 = int((y_center - box_height / 2) * img_h)
                    x2 = int((x_center + box_width / 2) * img_w)
                    y2 = int((y_center + box_height / 2) * img_h)

                    reference_bounding_box = [x1, y1, x2, y2]
                    logging.info(
                        f"Found reference box {reference_bounding_box} in image {reference_image_path}."
                    )
                    break
        if reference_bounding_box:
            break

    if not reference_bounding_box:
        raise RuntimeError("No helmet (class 0) found in YOLO labels directory.")

    # STEP 2: Load SAM model and initialize the predictor
    sam_registry_key = "vit_h"
    sam_model = sam_model_registry[sam_registry_key](checkpoint=args.sam_weights)
    sam_model.to(compute_device)
    predictor = SamPredictor(sam_model)
    logging.info("Loaded Segment Anything Model (SAM) and configured predictor.")

    # STEP 3: Generate a mask for the reference helmet crop
    ref_bgr = cv2.imread(reference_image_path)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(ref_rgb)
    masks, _, _ = predictor.predict(
        box=np.array([reference_bounding_box]),
        multimask_output=False
    )
    ref_mask = masks[0]
    logging.info("Generated reference mask for the helmet crop.")

    # STEP 4: Compute the CLIP embedding for the reference crop
    ref_crop = (ref_rgb * ref_mask[:, :, None]).astype(np.uint8)
    pil_ref = Image.fromarray(ref_crop)
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=compute_device)
    clip_input = clip_preprocessor(pil_ref).unsqueeze(0).to(compute_device)
    with torch.no_grad():
        ref_emb = clip_model.encode_image(clip_input)
        ref_emb /= ref_emb.norm(dim=-1, keepdim=True)
    logging.info("Computed normalized CLIP embedding for reference helmet.")

    # STEP 5: Prepare automatic mask generator
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    logging.info("Initialized automatic mask generator for target images.")

    # STEP 6: Iterate through target images and compute similarity scores with tqdm
    gt_df = pd.read_csv(args.gt_csv)
    all_stats = []

    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Processing target images"):  # noqa: E203
        img_name = row['image_name']
        true_pres = int(row['helmet_presence'])
        true_cnt = int(row['helmet_count'])
        img_path = os.path.join(args.target_dir, img_name)

        if not os.path.exists(img_path):
            logging.warning(f"Target image {img_path} not found. Skipping.")
            continue

        tgt_bgr = cv2.imread(img_path)
        tgt_rgb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)
        masks_list = mask_generator.generate(tgt_rgb)

        sims = []
        for m_info in masks_list:
            seg = m_info['segmentation']
            if seg.sum() < args.min_area:
                continue

            ys, xs = np.where(seg)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            patch = (tgt_rgb[y_min:y_max+1, x_min:x_max+1] * seg[y_min:y_max+1, x_min:x_max+1, None]).astype(np.uint8)
            pil_patch = Image.fromarray(patch)
            inp_patch = clip_preprocessor(pil_patch).unsqueeze(0).to(compute_device)

            with torch.no_grad():
                emb = clip_model.encode_image(inp_patch)
                emb /= emb.norm(dim=-1, keepdim=True)
                sims.append((ref_emb @ emb.T).item())

        all_stats.append({
            'image_name': img_name,
            'gt_presence': true_pres,
            'gt_count': true_cnt,
            'similarities': sims
        })

    # STEP 7: Compute overall ROC AUC
    gt_pres_list = [s['gt_presence'] for s in all_stats]
    max_sims_list = [max(s['similarities']) if s['similarities'] else 0.0 for s in all_stats]
    roc_auc_val = roc_auc_score(gt_pres_list, max_sims_list) if len(set(gt_pres_list)) > 1 else float('nan')
    logging.info(f"Overall ROC AUC: {roc_auc_val:.4f}")

    # STEP 8: Sweep thresholds and calculate metrics with tqdm
    thresholds = np.arange(
        args.thresh_start,
        args.thresh_end + args.thresh_step / 2,
        args.thresh_step
    )
    metrics_rows = []
    gt_counts = [s['gt_count'] for s in all_stats]

    for t in tqdm(thresholds, desc="Threshold sweep"):  # noqa: E203
        pres_preds, cnt_preds = [], []
        for s in all_stats:
            cnt_pred = sum(1 for sim in s['similarities'] if sim >= t)
            pres_preds.append(1 if cnt_pred > 0 else 0)
            cnt_preds.append(cnt_pred)

        metrics_rows.append({
            'threshold': t,
            'accuracy': accuracy_score(gt_pres_list, pres_preds),
            'precision': precision_score(gt_pres_list, pres_preds, zero_division=0),
            'recall': recall_score(gt_pres_list, pres_preds, zero_division=0),
            'f1_score': f1_score(gt_pres_list, pres_preds, zero_division=0),
            'roc_auc': roc_auc_val,
            'mse': mean_squared_error(gt_counts, cnt_preds),
            'mae': mean_absolute_error(gt_counts, cnt_preds),
            'rmse': np.sqrt(mean_squared_error(gt_counts, cnt_preds)),
            'count_accuracy': np.mean([g == p for g, p in zip(gt_counts, cnt_preds)]),
            'correct_preds': sum(g == p for g, p in zip(gt_counts, cnt_preds)),
            'under_preds': sum(p < g for g, p in zip(gt_counts, cnt_preds)),
            'over_preds': sum(p > g for g, p in zip(gt_counts, cnt_preds))
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved detailed metrics to {args.output_csv}.")

    # STEP 9: Generate and save or display metric plots
    plt.figure(figsize=(10, 7))
    for m in ['accuracy', 'precision', 'recall', 'f1_score', 'count_accuracy']:
        plt.plot(metrics_df['threshold'], metrics_df[m], label=m)
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Similarity Threshold')
    plt.legend()

    if args.save_plots:
        plt.savefig(args.plot_file)
        logging.info(f"Saved performance plot to {args.plot_file}.")
    else:
        plt.show()


if __name__ == '__main__':
    main()
