from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.track(source=0, show=True, tracker="bytetrack.yaml")