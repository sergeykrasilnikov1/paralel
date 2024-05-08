import argparse
import time
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import cv2
from multiprocessing import Pool, cpu_count

model = YOLO('yolov8n-pose.pt') 

def process_frame(frame):
    try:
        processed_frame = model.predict(frame, verbose=False)
        return processed_frame[0].plot()
    except Exception as e:
        print("Произошла ошибка:", e)
        return frame



def process_video_multithread(input_path, output_path, multithread):
    num_processes = cpu_count()
    print(num_processes)
    start_time = time.time()
    frames = []
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640,480))
    if not cap.isOpened():
        print("Ошибка: Невозможно открыть видеофайл.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (640, 480)))

    cap.release()
    frames = frames
    if multithread:
      with Pool(num_processes) as pool:
          processed_frames = pool.map(process_frame, frames)

      
      for frame in processed_frames:
          out.write(frame)

    else:
          for frame in frames:
              out.write(process_frame(frame))

    out.release()

    print("Время выполнения:", time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для ускорения инференса модели 'yolov8s-pose' на CPU.")
    parser.add_argument("input_video", help="Путь к входному видеофайлу (разрешение 640x480).")
    parser.add_argument("--multithread", action="store_true", help="Использовать многопоточность для ускорения обработки видео.")
    parser.add_argument("output_video", help="Имя выходного видеофайла.")
    args = parser.parse_args()
    process_video_multithread(args.input_video, args.output_video, args.multithread)

