# Real-time Sign Language Recognition

<!-- ![Demo](https://miro.medium.com/v2/resize:fit:1400/1*986e-bB7K39D4S-S70K4Yg.png) -->

Proyek ini adalah sistem pengenalan bahasa isyarat secara real-time menggunakan model deep learning YOLOv11. Sistem ini mendeteksi isyarat tangan dari stream kamera, menerjemahkannya menjadi teks, dan mengucapkan kata yang terdeteksi. Model ini dilatih pada 19 isyarat tangan umum.

## Fitur Utama

- **Deteksi Real-time**: Mampu mendeteksi isyarat tangan dari webcam dengan performa cepat.
- **Akurasi Tinggi**: Model mencapai akurasi mAP50 0.995 pada data validasi.
- **Output Teks dan Suara**: Kata yang terdeteksi ditampilkan di terminal dan diucapkan melalui fitur Text-to-Speech (TTS).
- **Efisiensi CPU**: Model YOLOv11 Nano dioptimalkan untuk inferensi di CPU, memastikan performa baik tanpa GPU yang kuat.

## Teknologi yang Digunakan

- **Model**: YOLOv11n (dilatih ulang dengan custom dataset)
- **Kerangka Kerja**: Ultralytics
- **Pustaka Inti**: PyTorch, OpenCV
- **Fitur Tambahan**: pyttsx3 (Text-to-Speech)

## Memulai Proyek

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di komputer Anda.

### 1. Persyaratan Sistem

- Python 3.10 atau versi lebih baru  
- Lingkungan virtual (venv atau conda)  
- Kamera (webcam)  
- Untuk training: VGA NVIDIA dengan CUDA Toolkit

### 2. Instalasi Dependensi

Sangat disarankan untuk membuat lingkungan virtual (venv) untuk mengisolasi paket.


```bash
# Buat dan aktifkan lingkungan virtual
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/MacOS
```
Tergantung pada kebutuhan Anda, instal dependensi yang sesuai
- Untuk Melatih model (Training) saya menggunakan ini :
```bash
pip install -r reuirement-train.txt
```
- Untuk Melatih Deteksi (Inference) uji coba saya menggunakan ini :
```bash
pip install -r reuirement-detect.txt
```


### 3. Persiapan Dataset dan Model

1. **Dataset**: Pastikan dataset Anda memiliki struktur folder yang benar (`Dataset/images`, `Dataset/labels`, dan `Dataset/dataset.yaml`).

   👉 [Klik di sini untuk mengunduh dataset dari Google Drive](https://drive.google.com/file/d/ID-GOOGLE-DRIVE/view?usp=sharing)


2. **Model**: Letakan file model misal yolo11n.pt di folder utama proyek Anda untuk proses training.

## File Model
| Model          | Size          |Map 50-90      |
|----------------|---------------|---------------|
| [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)  | 640           | 39.5          |
| [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)  | 640           | 47.0          |
| [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt)  | 640           | 51.5          |
| [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt)  | 640           | 53.4          |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)  | 640           | 54.7          |
 
Pelatihan dilakukan di laptop dengan spesifikasi berikut:

- **Prosesor**: Intel Core i5-1135G7 @ 2.40GHz
- **RAM**: 8GB DDR4
- **GPU**: Integrated Intel Iris Xe (tanpa GPU NVIDIA)
- **OS**: Windows 11 64-bit
- **Python**: 3.10 (virtualenv)

### Melatih Model
jalankan skrip training.py untuk memulai training. Pastikan lingkungan Anda aktif dan memiliki depedensi training.
```bash
python training.py
```
catatan: proses ini akan memakan waktu. Hasil model (best.pt) akan di simpan di folder model/


## Hasil Training Model
Model ini dilatih selama 100 epoch dan mencapai metrik berikut:

| Model          | mAP50(B)          |mAP 50-90      |
|----------------|-------------------|---------------|
| [YOLO11n](https://drive.google.com/drive/folders/1igH2g6rC4kupeYrRv4M63ipEvwhTqbxH?usp=sharing)  | 0.99495               | 0.94403          |
| [YOLO11s](https://drive.google.com/drive/folders/1RA6GSKWcQ9-U5al4ZRqR6qAyPlwHMzwA?usp=sharing)  | 0.99500               | 0.95843          |
| [YOLO11m](https://drive.google.com/drive/folders/1vlH1LevOJnmfSrc_HuZbyL2gmOvze0Rn?usp=sharing)  | 0.99500               | 0.96327          |
| [YOLO11l](https://drive.google.com/drive/folders/1NvdmNEeRLTnoM2sTLcvsvz7QMCPQMvZP?usp=sharing)  | 0.99500               | 0.95676          |
| [YOLO11x](https://drive.google.com/drive/folders/1TKhX1cRTSHjs4Yd5_QKKy4A_5oybQ5f-?usp=sharing)  | 0.99500               | 0.96327          |

### Menjalankan Deteksi Real-time
Setelah training selesai dan model best.pt sudah dihasilkan, jalankan skrip detect.py untuk memulai deteksi dari kamera. Pastikan Anda berada di lingkungan dengan dependensi inference.

```bash
python detect.py
```

### Hasil Deteksi Beberapa Model di CPU

Pengujian dilakukan di laptop dengan spesifikasi berikut:

- **Prosesor**: Intel(R) Core(TM) i3-1005G1 CPU @ 1.20GHz (1.20 GHz)
- **RAM**: 12.0 GB DDR4
- **GPU**: Tanpa GPU 
- **OS**: Windows 11 64-bit
- **Python**: 3.10 (virtualenv)

| Model     | Keterangan                  |
|-----------|-----------------------------|
| YOLOv11n  | Paling ringan, sangat cepat |
| YOLOv11s  | Cukup cepat dan stabil      |
| YOLOv11m  | Lumayan, mulai melambat     |
| YOLOv11l  | Cenderung berat di CPU      |
| YOLOv11x  | Terlalu berat tanpa GPU     |

> ⚠️ FPS diukur saat menjalankan `detect.py` secara real-time menggunakan webcam internal. Tidak menggunakan GPU.
<img width="1920" height="1080" alt="Screenshot (1660)" src="https://github.com/user-attachments/assets/9a678b9a-8d0e-4825-a7ff-cc1fe2c32d37" />
<img width="1919" height="1079" alt="Screenshot 2025-09-03 132346" src="https://github.com/user-attachments/assets/1f98d585-a6e2-45e9-a0d2-11fe79e01cdb" />
<img width="1920" height="1080" alt="Screenshot (1657)" src="https://github.com/user-attachments/assets/154b5a85-bda5-48a9-9899-9ce61043edf0" />
<img width="1920" height="1080" alt="Screenshot (1658)" src="https://github.com/user-attachments/assets/ce3db550-61e6-43bd-8be1-9302a884a260" />
<img width="1920" height="1080" alt="Screenshot (1659)" src="https://github.com/user-attachments/assets/df40106f-f648-4400-9775-03d999309cb9" />


## Struktur Proyek
```bash
sign_language_project/
├── Dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml
├── model/
│   └── model_yolov11m/
│       └── weights/
│           └── best.pt

├── training.py
├── detect.py
├── convert.py
├── requirements-train.txt
└── requirements-detect.txt
```

Author :  [HAMDANDIH](https://github.com/dansecret)
