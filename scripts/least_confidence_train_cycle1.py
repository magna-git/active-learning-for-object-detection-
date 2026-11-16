from pathlib import Path
from ultralytics import YOLO

ROOT = Path.cwd()

def main():
    # Reprendre le modèle initial entraîné sur 20 images
    model = YOLO(str(ROOT / "runs" / "train_initial" / "weights" / "best.pt"))

    # Entraîner avec les 36 images Least Confidence
    model.train(
        data=str(ROOT / "data_least_confidence.yaml"),
        epochs=15,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(ROOT / "runs"),
        name="least_confidence_cycle1",
        seed=42
    )

if __name__ == "__main__":
    main()