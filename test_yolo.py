from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(
    source="Test_install_YoloV8/Test_install_YoloV8/CirculationParis.mp4",
    conf=0.2,
    save=True,
    show=True,
    device=0,
    project="runs",
    name="video_paris"
)