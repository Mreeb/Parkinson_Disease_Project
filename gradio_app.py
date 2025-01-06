import os
from ultralytics import YOLO
import pandas as pd
import gradio as gr
from PIL import Image

PREDICTION_DIR = "Detection"  # YOLO prediction directory
os.makedirs(PREDICTION_DIR, exist_ok=True)

MODEL_PATH = "parkinson_detection.pt"

# Load the YOLO model
model = YOLO(MODEL_PATH)

def get_severity(confidence):
    """Classify confidence into severity levels."""
    if confidence < 0.5:
        return "Mild"
    elif confidence < 0.8:
        return "Moderate"
    else:
        return "Severe"

def predict_and_display(image):
    """Process the uploaded image, run YOLO model, and return results."""
    results = model.predict(source=image, save=True, conf=0.5, line_width=1, project=PREDICTION_DIR)

    latest_dir = max([os.path.join(PREDICTION_DIR, d) for d in os.listdir(PREDICTION_DIR)], key=os.path.getmtime)
    predicted_image_path = next(
        (os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), 
        None
    )

    detection_info = []
    parkinson_count = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())  # Get class ID (e.g., 0 for "Normal", 1 for "Parkinson")
        class_name = results[0].names[class_id]

        if class_name.lower() == "parkinson":
            severity = get_severity(confidence)
            parkinson_count += 1
        else:
            severity = "N/A"

        detection_info.append({
            "x1": round(x1),
            "y1": round(y1),
            "x2": round(x2),
            "y2": round(y2),
            "confidence": round(confidence, 2),
            "class": class_name,
            "severity": severity,
        })

    detection_summary = pd.DataFrame(detection_info)

    summary_text = (
        f"Total Detections: {len(detection_info)}\n"
        f"Parkinson Cases: {parkinson_count}"
    )

    map50_Score_path = "Logs/map50_Score.png"
    map50_95_Score_path = "Logs/map50-95_Score.png"
    precision_Score_path = "Logs/Precision_Score.png"
    recall_Score_path = "Logs/Recall_Score.png"

    return (
        Image.open(predicted_image_path),
        summary_text,
        detection_summary,
        Image.open(map50_Score_path),
        Image.open(map50_95_Score_path),
        Image.open(precision_Score_path),
        Image.open(recall_Score_path),
    )

with gr.Blocks() as app:
    gr.Markdown("# Parkinson's Disease Detection App")
    gr.Markdown("Upload an image to detect signs of Parkinson's disease.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image")

            detect_button = gr.Button("Detect")

        with gr.Column():
            output_image = gr.Image(label="Predicted Image")

    summary_output = gr.Text(label="Detection Summary")
    csv_output = gr.DataFrame(label="Detection Details")

    gr.Markdown("# Models Performance Metrics.")

    with gr.Row():
        with gr.Column():
            map50_Score = gr.Image(label="Map50 Score")
            map50_95_Score = gr.Image(label="Map50-95 Score")

        with gr.Column():
            precision_Score = gr.Image(label="Precision Score")
            recall_Score = gr.Image(label="Recall Score")

    # Button action to process image
    detect_button.click(
        predict_and_display, 
        inputs=image_input, 
        outputs=[
            output_image,
            summary_output,
            csv_output,
            map50_Score,
            map50_95_Score,
            precision_Score,
            recall_Score
        ]
    )

app.launch()
