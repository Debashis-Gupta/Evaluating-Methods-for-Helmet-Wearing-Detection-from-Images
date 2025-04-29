from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
import random
import datetime
import shutil
from ultralytics import YOLO

from tqdm import tqdm
from sklearn.model_selection import KFold
random.seed(0)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Add debugging to check directory structure
dataset_path = Path("/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Data")
print(f"Dataset path exists: {dataset_path.exists()}")
print(f"Directory contents: {list(dataset_path.iterdir())}")

# Fix the path to find labels in train and val subdirectories
labels = []
for subfolder in ["train", "val"]:
    subfolder_path = dataset_path / "labels" / subfolder
    if subfolder_path.exists():
        print(f"Searching in {subfolder_path}")
        labels.extend(sorted(subfolder_path.glob("*.txt")))
    else:
        print(f"Warning: {subfolder_path} does not exist")

print(f"Number of label files found: {len(labels)}")
if labels:
    print(f"First few labels: {labels[:5]}")
else:
    print("No label files found!")

yaml_file = "/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Project/Final_Code/config/helmet.yaml"
with open(yaml_file, encoding="utf8") as y:
    config = yaml.safe_load(y)
    classes = config["names"]
cls_idx = sorted(classes.keys())
print(cls_idx)
index = [label.stem for label in labels]  # uses base filename as ID (no extension)
labels_df = pd.DataFrame([], columns=cls_idx, index=index)

for label in labels:
    lbl_counter = Counter()

    with open(label) as lf:
        lines = lf.readlines()

    for line in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(line.split(" ")[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results
print(labels_df)
kfolds = list(kf.split(labels_df))

folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=index, columns=folds)

for i, (train, val) in enumerate(kfolds, start=1):
    folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
    folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

supported_extensions = [".jpg", ".jpeg", ".png"]

# Initialize an empty list to store image file paths
images = []

# Loop through supported extensions and gather image files
for ext in supported_extensions:
    images.extend(sorted((dataset_path / "images").glob(f"*{ext}")))

# Also check if images might be in subdirectories
if len(images) == 0:
    print("Trying to find images in subdirectories...")
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))
    print(f"Found {len(images)} images in subdirectories")

# Verify we have matching labels and images
image_stems = {img.stem for img in images}
label_stems = {lbl.stem for lbl in labels}
print(f"Number of images found: {len(images)}")
print(f"Number of labels with matching images: {len(image_stems & label_stems)}")
print(f"Number of labels without matching images: {len(label_stems - image_stems)}")
print(f"Number of images without matching labels: {len(image_stems - label_stems)}")

# Create the necessary directories and dataset YAML files
save_path = Path(dataset_path / f"{timestamp}_{config['model_name'].split('.')[0]}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)



    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "train",
                "val": "val",
                "names": classes,
            },
            ds_y,
        )


# Create a mapping from label stems to full paths
label_dict = {label.stem: label for label in labels}

for image in tqdm(images, total=len(images), desc="Copying files"):
    # Only process if there's a matching label
    if image.stem in label_dict:
        label = label_dict[image.stem]
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            try:
                shutil.copy(image, img_to_path / image.name)
                shutil.copy(label, lbl_to_path / label.name)
            except shutil.SameFileError:
                pass  # File already exists, skip
            except Exception as e:
                print(f"Error copying {image.name}: {e}")

folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

results = {}

# Define your additional arguments here
batch = config.get('batch_size', 16)  # Get batch size from config or use default
project = config['project_name']
epochs = config.get('epochs', 100)  # Get epochs from config or use default

for k, dataset_yaml in enumerate(ds_yamls):
    model = YOLO(config['model_name'], task="detect")
    results[k] = model.train(
        data=dataset_yaml, 
        epochs=epochs, 
        batch=batch, 
        project=project, 
        name=f"fold_{k + 1}",
        imgsz=config['img_size']
    )  # Use 'batch' instead of 'batch_size' as per error message