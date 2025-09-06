import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time

model = YOLO('model/model_yolov11n/weights/best.pt')

class_names = [
    'bis', 'halo', 'kapan', 'maaf', 'makan', 'minum', 'sama-sama', 'semangat', 'telfon', 'terimakasih',
]

THRESHOLD_SAFE = 7
counter_safety = 0
safety_label = "none"


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
                    if not engine.isBusy():
                        threading.Thread(target=speak, args=(current_label,)).start()
                    last_spoken_label = current_label

        else:
            last_spoken_label = ""
            counter_safety = 0



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
