
from pathlib import Path
import random
import shutil

# 1. Locate the unlabeled image pool
ROOT = Path.cwd()
pool_dir = ROOT / "dataset" / "images" / "pool"

# 2. Random sampling parameters
SEED = 45
B = 16

# 3. Select 16 random images without duplicates
pool_images = sorted(pool_dir.glob("*.jpg"))
rng = random.Random(SEED)
selected = rng.sample(pool_images, B)

print("Pool before:", len(pool_images))
print("Selected:", len(selected))

# 4. Move selected images and reveal their labels
for image in selected:
    label = ROOT / "dataset" / "oracle_labels" / "pool" / (image.stem + ".txt")

    shutil.move(str(image), str(ROOT / "dataset" / "images" / "train"))
    shutil.move(str(label), str(ROOT / "dataset" / "labels" / "train"))

# 5. Verify the new dataset sizes
print("Train:", len(list((ROOT / "dataset" / "images" / "train").glob("*.jpg"))))
print("Pool:", len(list(pool_dir.glob("*.jpg"))))
