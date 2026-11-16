
from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # Train on the initial 20 labeled images
    model.train(
        data="data.yaml",
        epochs=15,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project="C:/Users/khadijetou/Desktop/esigelec/year2026_2027/dossier_active_learning/runs",
        name="train_initial"
    )

# Entry point: safe execution on Windows
if __name__ == "__main__":
    main()
