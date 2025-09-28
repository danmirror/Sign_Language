import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time

import socket
import json
import time
import random
import struct

# --- Konfigurasi UDP ---
# UDP_IP = "127.0.0.1"
UDP_IP = "103.151.141.219"
UDP_PORT_VIDEO = 4000
UDP_PORT_DATA = 5000

sock_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video.connect((UDP_IP, UDP_PORT_VIDEO))
model = YOLO('model/model_yolov11n/weights/best.pt')

class_names = [
    'bis', 'halo', 'kapan', 'maaf', 'makan', 'minum', 'sama-sama', 'semangat', 'telfon', 'terimakasih',
]

THRESHOLD_SAFE = 7
counter_safety = 0
clases_detection = "none"
safety_label = "none"
is_detection = False


# Inisialisasi Text-to-Speech (TTS)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.say("Aplikasi siap digunakan.")
engine.runAndWait()

# Kunci untuk sinkronisasi thread
speak_lock = threading.Lock()

def speak(text):
    with speak_lock:
        engine.say(text)
        engine.runAndWait()


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Tidak bisa membuka stream kamera.")
    exit()

print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")
last_spoken_label = ""
prev_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame dari kamera.")
            break
        # frame = cv2.resize(frame, (640, 480))

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        results = model.predict(source=frame, conf=0.25, verbose=False, device='cpu')
        
        best_box = None
        best_confidence = 0.0

       

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = box

        if best_box and best_confidence > 0.7:
            class_id = int(best_box.cls[0])
            label_text = f"{class_names[class_id]} {best_confidence:.2f}"
            
            x1, y1, x2, y2 = best_box.xyxy[0].int().tolist()
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            current_label = class_names[class_id]
            if current_label != last_spoken_label:
                counter_safety +=1

                if safety_label != class_names[class_id]:
                    safety_label = class_names[class_id]
                    counter_safety = 0


                print(f"Terdeteksi: {current_label}, counter_safety: {counter_safety}")
                if(counter_safety > THRESHOLD_SAFE):
                    is_detection = True
                    clases_detection = class_names[class_id]

                    if not engine.isBusy():
                        threading.Thread(target=speak, args=(current_label,)).start()
                    last_spoken_label = current_label
            
                
                
        else:
            last_spoken_label = ""
            counter_safety = 0

        # encode ke JPEG
        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
        if ret:
            data_frame = buf.tobytes()
            # kirim panjang frame dulu
            sock_video.sendall(struct.pack("!I", len(data_frame)))
            # kirim data frame
            sock_video.sendall(data_frame)
                
        # buat data dummy 4 angka sensor
        sensor_data = {
            "val1": is_detection,
            "val2": clases_detection,
            "val3": counter_safety,
            "val4": THRESHOLD_SAFE,
        }

        # kirim JSON ke server Go
        sock_data.sendto(json.dumps(sensor_data).encode(), (UDP_IP, UDP_PORT_DATA))



        cv2.imshow('Real-time Sign Language Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        engine.stop()
    except Exception:
        pass
