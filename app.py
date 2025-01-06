from ultralytics import YOLO

model = YOLO("parkinson_detection.pt")

results = model.predict("img.jpg", save=True)

print(results)