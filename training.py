import os
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    # Muat model YOLOv11 Nano yang sudah dilatih
    model = YOLO('yolo11s.pt')

    # Jalankan training dengan konfigurasi yang lebih baik
    results = model.train(
        data='Dataset/dataset.yaml',  
        epochs=100,                   
        imgsz=640,                    
        batch=16,                     
        device='0',                   # Gunakan GPU 0
        project='model',
        name='model_yolov11x',
        # Tambahkan workers untuk pemrosesan data paralel
        workers=4,
        # Mengaktifkan data augmentation
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
        translate=0.1,
        degrees=0.1
    )

    print("Training selesai. Model terbaik disimpan di: model")