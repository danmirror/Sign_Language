import cv2
from ultralytics import YOLO
import pyttsx3
import threading

# Muat model YOLO yang sudah dilatih
model = YOLO('model/yolov11s.pt')

# Inisialisasi daftar nama kelas
class_names = [
    'benar', 'bertemu', 'bis', 'coba', 'halo', 'kamu', 'kapan', 'kereta',
    'maaf', 'makan', 'minum', 'mobil', 'motor', 'sama-sama', 'sekarang', 
    'semangat', 'telefon', 'terimakasih', 'tidur', 'toilet'
]

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.say("Aplikasi siap digunakan.")
engine.runAndWait()

speak_lock = threading.Lock()

# Fungsi untuk mengucapkan teks dengan kunci
def speak(text):
    with speak_lock:
        engine.say(text)
        engine.runAndWait()

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Tidak bisa membuka stream kamera.")
    exit()

print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")
last_spoken_label = "" # Variabel untuk menghindari pengucapan berulang

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame dari kamera.")
            break

        results = model.predict(source=frame, conf=0.6, verbose=False, device='cpu')
        
        detected_labels = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if confidence > 0.7:
                    detected_labels.append(class_names[class_id])

                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                label_text = f"{class_names[class_id]} {confidence:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if detected_labels:
            current_label = max(set(detected_labels), key=detected_labels.count)
            if current_label != last_spoken_label:
                print(f"Terdeteksi: {current_label}")
                # Jalankan pengucapan di thread baru
                threading.Thread(target=speak, args=(current_label,)).start()
                last_spoken_label = current_label
        else:
            last_spoken_label = ""

        cv2.imshow('Real-time Sign Language Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()