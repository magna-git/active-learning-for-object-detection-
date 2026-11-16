from pathlib import Path
from ultralytics import YOLO

ROOT = Path.cwd()

def main():
    # Charger le meilleur modèle du cycle 2
    model = YOLO(str(ROOT / "runs" / "random_cycle2" / "weights" / "best.pt"))

    # Réentraîner avec les 68 images
    model.train(
        data=str(ROOT / "data.yaml"),
        epochs=15,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(ROOT / "runs"),
        name="random_cycle3",
        seed=42
    )

if __name__ == "__main__":
    main()