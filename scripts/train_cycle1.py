from pathlib import Path
from ultralytics import YOLO

ROOT = Path.cwd()

def main():
    # Load the best model trained on 20 images
    model = YOLO(str(ROOT / "runs" / "train_initial" / "weights" / "best.pt"))

    # Continue training with 36 images
    model.train(
        data=str(ROOT / "data.yaml"),
        epochs=15,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(ROOT / "runs"),
        name="random_cycle1",
        seed=42
    )

if __name__ == "__main__":
    main()