import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

# =========================
# LOAD MODEL
# =========================

model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='best.pt',
    force_reload=False,
    trust_repo=True
)

model.conf = 0.25
model.iou = 0.45

# =========================
# CLASS NAMES
# =========================

CAR_CLASSES = {
    0: "audi",
    1: "bmw",
    2: "ferrari",
    3: "ford",
    4: "honda",
    5: "toyota",
    6: "hyundai",
    7: "nissan",
    8: "jeep",
    9: "astonmartin"
}

CONFIDENCE_THRESHOLD = 0.30

# =========================
# IMAGE DETECTION
# =========================

def detect_image(img):

    img_array = np.array(img)

    results = model(img_array)

    for det in results.xyxy[0]:

        x1, y1, x2, y2 = map(int, det[:4])
        confidence = float(det[4])
        class_id = int(det[5])

        if confidence < CONFIDENCE_THRESHOLD:
            label = f"Unknown Vehicle ({confidence:.0%})"
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
            0.6,
            color,
            2
        )

    return Image.fromarray(img_array)

# =========================
# VIDEO DETECTION
# =========================

def detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        for det in results.xyxy[0]:

            x1, y1, x2, y2 = map(int, det[:4])

            confidence = float(det[4])
            class_id = int(det[5])

            if confidence < CONFIDENCE_THRESHOLD:
                label = f"Unknown Vehicle ({confidence:.0%})"
                color = (255, 0, 0)

            else:
                label = f"{CAR_CLASSES[class_id]} ({confidence:.0%})"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        out.write(frame)

    cap.release()
    out.release()

    return output_path

# =========================
# GRADIO UI
# =========================

with gr.Blocks() as app:

    gr.Markdown("# 🚗 AI Car Brand Detector")

    with gr.Tab("Image Detection"):

        gr.Interface(
            fn=detect_image,
            inputs=gr.Image(type="pil"),
            outputs=gr.Image(type="pil"),
            title="Upload Car Image"
        )

    with gr.Tab("Video Detection"):

        gr.Interface(
            fn=detect_video,
            inputs=gr.Video(),
            outputs=gr.Video(),
            title="Upload Car Video"
        )

app.launch()