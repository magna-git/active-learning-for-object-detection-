
from pathlib import Path
import random
import shutil

# 1. CONFIGURATION: define dataset sizes and random seed
SEED = 42
train_size = 20
pool_size = 66
val_size = 20
test_size = 20

# 2. PATHS: locate the original COCO128 dataset
ROOT = Path.cwd()
dataset = ROOT / "raw_coco128-20261110T083254Z-1-001" / "raw_coco128" / "coco128"

images_dir = dataset / "images" / "train2017"
labels_dir = dataset / "labels" / "train2017"

# Check that source folders exist
assert images_dir.exists(), "Images folder not found"
assert labels_dir.exists(), "Labels folder not found"

# 3. LOAD: collect image files
images = sorted(images_dir.glob("*.jpg"))
print("Number of images:", len(images))

# 4. VALIDATE: match each image with its annotation
pairs = []

for image in images:
    label = labels_dir / (image.stem + ".txt")

    if label.exists():
        pairs.append((image, label))

print("Valid pairs:", len(pairs))

# 5. SHUFFLE: same random split for every experiment
random.seed(SEED)
random.shuffle(pairs)

# 6. SPLIT: Train / Pool / Validation / Test
pairs_train = pairs[:train_size]

pairs_pool = pairs[train_size:train_size + pool_size]

pairs_val = pairs[
    train_size + pool_size:
    train_size + pool_size + val_size
]

pairs_test = pairs[
    train_size + pool_size + val_size:
]

print("Train:", len(pairs_train))
print("Pool:", len(pairs_pool))
print("Validation:", len(pairs_val))
print("Test:", len(pairs_test))

# 7. RESET: start each Active Learning strategy from cycle 0
dataset_dir = ROOT / "dataset_experiences" / "kmeans"

if dataset_dir.exists():
    shutil.rmtree(dataset_dir)

# 8. CREATE: prepare YOLO dataset folders
for split in ["train", "pool", "val", "test"]:
    (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Hide Pool annotations until they are selected
(dataset_dir / "oracle_labels" / "pool").mkdir(parents=True, exist_ok=True)

# 9. COPY: distribute images and annotations
groups = {
    "train": pairs_train,
    "pool": pairs_pool,
    "val": pairs_val,
    "test": pairs_test
}

for split, items in groups.items():

    for image, label in items:

        # Copy image to its corresponding dataset folder
        shutil.copy2(image, dataset_dir / "images" / split)

        # Pool labels are hidden from YOLO
        if split == "pool":
            shutil.copy2(label, dataset_dir / "oracle_labels" / "pool")
        else:
            shutil.copy2(label, dataset_dir / "labels" / split)

    print(split, ":", len(items), "pairs copied")

print("Dataset preparation completed!")
