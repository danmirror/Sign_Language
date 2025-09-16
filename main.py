from flask import Flask, request, Response
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("model/model_yolov11n/weights/best.pt")

class_names = [
    'bis', 'halo', 'kapan', 'maaf', 'makan', 'minum', 
    'sama-sama', 'semangat', 'telfon', 'terimakasih',
]

@app.route("/process", methods=["POST"])
def process():
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return "Failed to decode image", 400

    # Jalankan YOLO
    results = model.predict(source=img, conf=0.25, verbose=False, device="cpu")

    best_box = None
    best_confidence = 0.0
    best_label = ""

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence > best_confidence:
                best_confidence = confidence
                best_box = box
                best_label = class_names[int(box.cls[0])]

    # Kalau ada deteksi → gambar kotak
    if best_box and best_confidence > 0.7:
        x1, y1, x2, y2 = best_box.xyxy[0].int().tolist()
        label_text = f"{best_label} {best_confidence:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Encode balik ke JPEG
    _, buffer = cv2.imencode(".jpg", img)
    return Response(buffer.tobytes(), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
