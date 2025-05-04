#!/usr/bin/env python3

import sys
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO, RTDETR

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from typing import Dict, List
import csv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)

def load_ground_truth(csv_path: str) -> Dict[str, Dict[str, int]]:
    gt = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt[row['image_name']] = {
                    'helmet_presence': int(row['helmet_presence']),
                    'helmet_count':     int(row['helmet_count'])
                }
        print(f"Loaded {len(gt)} ground truth entries", flush=True)
    except Exception as e:
        print(f"Error loading ground truth CSV: {e}", flush=True)
    return gt

def calculate_metrics(
    ground_truth: Dict[str, Dict[str, int]],
    image_paths: List[str],
    y_pred_presence: List[int],
    y_pred_count:    List[int],
) -> Dict[str, float]:
    """
    Compare predictions to ground truth, returning:
      accuracy, precision, recall, f1_score,
      mse, mae, rmse,
      num_samples
    """
    y_true_p, y_true_c = [], []
    y_pred_p, y_pred_c = [], []

    for img_path, p, c in tqdm(
        zip(image_paths, y_pred_presence, y_pred_count),
        desc="Calculating metrics",
        unit="img",
        leave=False
    ):
        name = os.path.basename(img_path)
        if name in ground_truth:
            y_true_p.append(ground_truth[name]['helmet_presence'])
            y_pred_p.append(p)
            y_true_c.append(ground_truth[name]['helmet_count'])
            y_pred_c.append(c)

    if not y_true_p:
        return {}

    # classification
    accuracy  = accuracy_score(y_true_p, y_pred_p)
    precision = precision_score(y_true_p, y_pred_p, zero_division=0)
    recall    = recall_score(y_true_p, y_pred_p, zero_division=0)
    f1        = f1_score(y_true_p, y_pred_p, zero_division=0)
    # regression
    mse = mean_squared_error(y_true_c, y_pred_c)
    mae = mean_absolute_error(y_true_c, y_pred_c)
    rmse = np.sqrt(mse)

    return {
        'accuracy':   accuracy,
        'precision':  precision,
        'recall':     recall,
        'f1_score':   f1,
        'mse':        mse,
        'mae':        mae,
        'rmse':       rmse,
        'num_samples': len(y_true_p),
    }

def get_model_paths(model_name: str, base_dir: str = '../model') -> Dict[int, str]:
    folders = {
        'Yolov5':  'Helmet_YOLOv5n',
        'Yolov8':  'Helmet_YOLOv8n',
        'Yolov12': 'Helmet_YOLOv12n',
        'RTDETR':  'Helmet_RTDETR'
    }
    if model_name not in folders:
        raise ValueError(f"Unrecognized model_name: {model_name!r}")
    folder = folders[model_name]
    return {
        fold: os.path.join(base_dir, folder, f'fold_{fold}', 'weights', 'best.pt')
        for fold in range(1, 6)
    }

def load_model(model_path: str, model_name: str, is_rtdetr: bool=False):
    try:
        print(f"[load_model] Loading {model_name} from {model_path}...", flush=True)
        mdl = RTDETR(model_path) if (is_rtdetr or model_name=="RTDETR") else YOLO(model_path)
        print(f"[load_model] {model_name} loaded.", flush=True)
        return mdl
    except Exception as e:
        print(f"[load_model] Error loading {model_name}: {e}", flush=True)
        return None

def sweep_confidence(
    model_types: List[str],
    gt_csv: str,
    img_dir: str,
    conf_min: float,
    conf_max: float,
    conf_step: float,
    save_annotated: bool
) -> pd.DataFrame:
    confs = np.arange(conf_min, conf_max + 1e-8, conf_step)
    image_paths = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.png','jpg','jpeg','bmp','tiff'))
    ]
    ground_truth = load_ground_truth(gt_csv)
    records = []

    for model_name in model_types:
        print(f"\n=== {model_name}: preload folds ===")
        fold_paths = get_model_paths(model_name)
        models = {
            fold: load_model(path, model_name=model_name)
            for fold, path in fold_paths.items()
        }

        raw_results: Dict[int, Dict[str, np.ndarray]] = {}
        for fold, mdl in models.items():
            if mdl is None:
                continue
            print(f"[{model_name}] Fold {fold}: raw inference …")
            raw_results[fold] = {}
            for img in tqdm(image_paths, desc=f"Fold {fold}", unit="img", leave=False):
                res = mdl(
                    img,
                    conf=conf_min,
                    verbose=False, show=False, save=False
                )
                raw_results[fold][img] = res[0].boxes.conf.cpu().numpy()

        for conf in tqdm(confs, desc="Threshold sweep", unit="conf"):
            #print(f"\n→ {model_name} @ conf={conf:.2f}")
            per_fold_metrics = {}
            for fold, confs_arr in raw_results.items():
                preds_p, preds_c = [], []
                for img in image_paths:
                    n = int(np.sum(confs_arr[img] >= conf))
                    preds_p.append(1 if n>0 else 0)
                    preds_c.append(n)
                per_fold_metrics[fold] = calculate_metrics(
                    ground_truth, image_paths, preds_p, preds_c
                )
            # aggregate
            valid = [v for v in per_fold_metrics.values() if v]
            if not valid:
                continue
            keys = valid[0].keys()
            mean = {k: np.mean([m[k] for m in valid]) for k in keys}
            std  = {k: np.std([m[k] for m in valid])   for k in keys}

            rec = {'model': model_name, 'conf': conf}
            for k in mean:
                rec[k]        = mean[k]
                rec[f"{k}_std"] = std[k]
            records.append(rec)

    return pd.DataFrame.from_records(records)

def plot_metrics(df: pd.DataFrame, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    metrics = ['accuracy','precision','recall','f1_score','mse','mae','rmse']
    for m in metrics:
        plt.figure(figsize=(8,5))
        for model in df['model'].unique():
            sub = df[df['model']==model]
            plt.errorbar(
                sub['conf'], sub[m], yerr=sub[f"{m}_std"],
                label=model, marker='o', capsize=3
            )
        plt.title(f"{m.replace('_',' ').title()} vs. Confidence")
        plt.xlabel("Confidence")
        plt.ylabel(m.replace('_',' ').title())
        if m in ('mse','mae','rmse'):
            plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()
        out_f = os.path.join(plot_dir, f"{m}_vs_conf.png")
        plt.savefig(out_f, dpi=200)
        plt.close()
        print(f"  ↳ saved {out_f}")

def main():
    p = argparse.ArgumentParser(
        description='Sweep helmet-detection confidence thresholds'
    )
    p.add_argument('--model-type', nargs='+',
                   default=['Yolov5','Yolov8','Yolov12','RTDETR'])
    p.add_argument('-g','--ground-truth', required=True)
    p.add_argument('-i','--images',      required=True)
    p.add_argument('--conf-min',  type=float, required=True)
    p.add_argument('--conf-max',  type=float, required=True)
    p.add_argument('--conf-step', type=float, required=True)
    p.add_argument('--save-annotated', action='store_true')
    p.add_argument('--output-csv',    default='metrics_over_conf.csv')
    args = p.parse_args()

    df = sweep_confidence(
        args.model_type,
        args.ground_truth,
        args.images,
        args.conf_min,
        args.conf_max,
        args.conf_step,
        args.save_annotated
    )

    df.to_csv(args.output_csv, index=False)
    print(f"\nWritten combined metrics to {args.output_csv}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_folder = f"plots_{ts}"
    print(f"Saving plots into: {plot_folder}")
    plot_metrics(df, plot_folder)

if __name__ == "__main__":
    main()
