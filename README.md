# 🚗 AI Car Brand Detector

YOLOv5-based deep learning model that detects 10 different car brands and flags unknown vehicles separately.

---

## 📌 Features

* Detects 10 car brands
* Unknown vehicle handling using confidence threshold
* Image/Video upload interface using Gradio
* Real-time object detection
* Bounding box visualization

---

## 🧠 Classes

* Audi
* BMW
* Ferrari
* Ford
* Honda
* Toyota
* Hyundai
* Nissan
* Jeep
* Aston Martin

---

## 🛠️ Tech Stack

* YOLOv5
* PyTorch
* OpenCV
* Gradio
* Python

---

## 🚀 Run Locally

```bash
git clone https://github.com/Naazpathan111/car-detection-yolov5.git

cd car-detection-yolov5

pip install -r requirements.txt

python app1.py
```

---

## ⚠️ Known Limitations

* Rear-view accuracy is lower than front-view
* Unknown vehicles may sometimes map to closest known class
* Performance depends on image quality and lighting

---

## 📷 Demo

Upload a car image/video and the model predicts:

* car brand
* confidence score
* unknown vehicle detection

---

## 👨‍💻 Author

Naaz Pathan
