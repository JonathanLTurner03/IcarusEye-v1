from PyQt6.QtCore import pyqtSignal, QObject, QThread
from PyQt6.QtGui import QImage
import subprocess
import cv2
import re
import torch
from ultralytics import YOLO
import time
from threading import Thread


def is_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

def list_ffmpeg_devices():
    if not is_ffmpeg_installed():
        print("FFmpeg is not installed. Please install FFmpeg to use this function.")
        return {}

    devices = {}
    try:
        result = subprocess.run(
            ['ffmpeg', '-f', 'dshow', '-list_devices', 'true', '-i', 'dummy'],
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stderr
        device_pattern = re.compile(r'\[dshow @ [^]]+] "([^"]+)" \(video\)')
        device_names = device_pattern.findall(output)
        for index, name in enumerate(device_names):
            devices[index] = name
    except Exception as e:
        print(f"Error listing video capture devices with FFmpeg: {e}")

    return devices

def list_opencv_devices(max_devices=5, api=cv2.CAP_DSHOW):
    devices = []
    for device_index in range(max_devices):
        cap = cv2.VideoCapture(device_index, api)
        if cap.isOpened():
            devices.append(device_index)
            cap.release()
    return devices


class DeviceScanner(QObject):
    devices_scanned = pyqtSignal(list)

    def run(self):
        """Scan for available video input devices and their details."""
        device_info = []
        if is_ffmpeg_installed():
            ffmpeg_devices = list_ffmpeg_devices()
            opencv_devices = list_opencv_devices(max_devices=len(ffmpeg_devices))
            for index in opencv_devices:
                if index in ffmpeg_devices:
                    device = [index, ffmpeg_devices[index]]
                    device_info.append(device)
        else:
            for index in list_opencv_devices():
                device_info[index] = f"Device {index}"
        print(device_info)
        self.devices_scanned.emit(device_info)


class RenderProcessor(QThread):
    frame_updated = pyqtSignal(QImage)  # Signal to emit frames to the GUI

    def __init__(self, result_queue, model_names, fps_target=60):
        super().__init__()
        self.result_queue = result_queue
        self.model_names = model_names
        self.fps_target = fps_target
        self.frame_duration = 1.0 / fps_target
        self.running = True

    def run(self):
        while self.running:
            start_time = time.time()
            try:
                # Retrieve a frame and detection results
                frame, results = self.result_queue.get(timeout=1)
            except:
                continue

            # Draw bounding boxes and predictions on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = f"{self.model_names[cls]}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Convert frame to QImage and emit it
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_updated.emit(qt_image)

            # Control frame rate
            elapsed_time = time.time() - start_time
            if elapsed_time < self.frame_duration:
                time.sleep(self.frame_duration - elapsed_time)

    def stop(self):
        self.running = False
        self.wait()

class DetectionProcessor(Thread):
    def __init__(self, video_path, model_path, result_queue, batch_size=4):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.running = True
        self.result_queue = result_queue
        self.batch_size = batch_size

        # Load the YOLO model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

    def run(self):
        frames = []
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frames.append(frame)
            if len(frames) == self.batch_size:
                # Perform object detection on the batch
                results = self.model(frames)

                # Put the results in the result queue
                for frame, result in zip(frames, results):
                    self.result_queue.put((frame, result))

                frames = []

        # Process remaining frames if any
        if frames:
            results = self.model(frames)
            for frame, result in zip(frames, results):
                self.result_queue.put((frame, result))

    def stop(self):
        self.running = False
        self.cap.release()