import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

# LOAD MODEL
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='best.pt',
    trust_repo=True
)

# SETTINGS
model.conf = 0.25
model.iou = 0.45

# AUTO CLASS NAMES
CAR_CLASSES = model.names

CONFIDENCE_THRESHOLD = 0.30

def detect(img):

    img_array = np.array(img)

    results = model(img_array)

    for det in results.xyxy[0]:

        x1, y1, x2, y2 = map(int, det[:4])

        confidence = float(det[4])

        class_id = int(det[5])

        if confidence < CONFIDENCE_THRESHOLD:

            label = f"Unknown ({confidence:.0%})"
            color = (255, 0, 0)

        else:

            label = f"{CAR_CLASSES[class_id]} ({confidence:.0%})"
            color = (0, 255, 0)

        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            img_array,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return Image.fromarray(img_array)


# GRADIO UI
app = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="AI Car Brand Detector",
    description="Detects car brands using YOLOv5"
)

app.launch()
