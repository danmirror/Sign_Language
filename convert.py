import cv2
import os
import glob
import re

def process_video_folder(input_folder, output_base_folder, frame_interval=5):
    supported_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in supported_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not video_files:
        print("Tidak ada file video yang ditemukan di folder input.")
        return

    total_frames_extracted = 0

    for video_path in video_files:
        video_file_name = os.path.splitext(os.path.basename(video_path))[0]
        

        class_name = video_file_name.split('_')[0]

        output_class_folder = os.path.join(output_base_folder, class_name)
        os.makedirs(output_class_folder, exist_ok=True)

        video_capture = cv2.VideoCapture(video_path)
        
        if not video_capture.isOpened():
            print(f"Error: Tidak bisa membuka file video {video_file_name}.")
            continue

        frame_count = 0
        saved_frame_count = 0
        
        print(f"Mengkonversi '{video_file_name}' ke frame...")
        
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            
            if frame_count % frame_interval == 0:
                new_frame_name = f"{class_name}_{saved_frame_count:05d}.jpg"
                new_frame_path = os.path.join(output_class_folder, new_frame_name)
                
                cv2.imwrite(new_frame_path, frame)
                saved_frame_count += 1
                
            frame_count += 1
            
        video_capture.release()
        print(f"Selesai! Mengekstrak {saved_frame_count} frame dan disimpan di '{output_class_folder}'.")
        total_frames_extracted += saved_frame_count

    print(f"\n==========================================")
    print(f"Konversi selesai! Total {total_frames_extracted} frame berhasil diekstrak.")
    print("Folder output Anda sekarang terorganisir per kelas, siap untuk anotasi.")

input_videos_folder = 'Dataset/video/'
output_frames_folder = 'Dataset/images/'
frames_to_skip = 5 

process_video_folder(input_videos_folder, output_frames_folder, frames_to_skip)