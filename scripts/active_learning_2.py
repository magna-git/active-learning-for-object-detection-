from pathlib import Path
from ultralytics import YOLO
import shutil

ROOT = Path.cwd()

DATASET = ROOT / "dataset_experiences" / "least_confidence"
pool_dir = DATASET / "images" / "pool"

# Modèle initial : entraîné sur les 20 premières images
#odel = YOLO(str(ROOT / "runs" / "train_initial" / "weights" / "best.pt")) for the round 1

model = YOLO(str(ROOT / "runs" / "least_confidence_cycle1-2" / "weights" / "best.pt"))
scores = []

# Analyser toutes les images du Pool
for image in sorted(pool_dir.glob("*.jpg")):

    results = model.predict(
        source=str(image),
        conf=0.05,
        device=0,
        verbose=False
    )

    confidences = results[0].boxes.conf.cpu().numpy()

    if len(confidences) > 0:
        score = float(confidences.min())
    else:
        score = 1.0

    scores.append((image, score))

# Classer les images de la moins confiante à la plus confiante
scores.sort(key=lambda x: (x[1], x[0].name))

# Sélectionner les 16 premières
selected = scores[:16]

print("Images analysées :", len(scores))
print("Images sélectionnées :", len(selected))

for image, score in selected:
    print(image.name, ":", round(score, 4))

for image, score in selected:

    label = DATASET / "oracle_labels" / "pool" / (image.stem + ".txt")

    shutil.move(
        str(image),
        str(DATASET / "images" / "train")
    )

    shutil.move(
        str(label),
        str(DATASET / "labels" / "train")
    )

print("Train :", len(list((DATASET / "images" / "train").glob("*.jpg"))))
print("Pool :", len(list(pool_dir.glob("*.jpg"))))