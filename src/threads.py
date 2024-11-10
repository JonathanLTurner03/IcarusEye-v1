import numpy as np
from PyQt6.QtCore import pyqtSignal, QObject, QThread
from PyQt6.QtGui import QImage
import subprocess
import cv2
import re
import torch
from ultralytics import YOLO
import time
from threading import Thread, Lock


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
    frame_updated = pyqtSignal(np.ndarray)  # Signal to emit frames to the GUI
    fps_updated = pyqtSignal(float)  # Signal to emit the FPS to the GUI

    def __init__(self, result_queue, model_names, fps_target=60):
        super().__init__()
        self.result_queue = result_queue
        self.model_names = model_names
        self.fps_target = fps_target
        self.frame_duration = 1.0 / fps_target
        self.running = True
        self.alive = True
        self.frame_times = []
        self.conf_thres = 0.5
        self.toggle_color_map(False)

    def run(self):
        while self.alive:
            while self.running:
                start_time = time.time()

                # Retrieve a frame and detection results
                try:
                    frame, results = self.result_queue.get(timeout=1)
                except:
                    print("No frame in queue; continuing...")
                    continue  # Skip processing if no frame is available

                # Draw bounding boxes and predictions on the frame
                try:
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf.item()
                            cls = int(box.cls.item())
                            label = f"{self.model_names[cls]}: {conf:.2f}"
                            if conf > self.conf_thres:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_map[cls], 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            self.color_map[cls], 2)

                    # Emit the processed frame as a numpy array
                    self.frame_updated.emit(frame)
                except Exception as e:
                    print(f"Error updating frame: {e}")

                # Calculate and emit FPS
                elapsed_time = time.time() - start_time
                self.frame_times.append(elapsed_time)

                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)

                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                self.fps_updated.emit(current_fps)

                # Enforce frame rate limit
                if elapsed_time < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed_time)

    def toggle_color_map(self, value):
        if value:
            self.color_map = \
                {
                    0: (255, 0, 0),
                    1: (255, 0, 0),
                    2: (0, 255, 0),
                    3: (0, 0, 255),
                    4: (255, 255, 0),
                    5: (255, 0, 255),
                    6: (0, 255, 255),
                    7: (255, 255, 255),
                    8: (255, 255, 255),
                    9: (255, 255, 255),
                }
        else:
            self.color_map = \
                {
                    0: (0, 255, 0),
                    1: (0, 255, 0),
                    2: (0, 255, 0),
                    3: (0, 255, 0),
                    4: (0, 255, 0),
                    5: (0, 255, 0),
                    6: (0, 255, 0),
                    7: (0, 255, 0),
                    8: (0, 255, 0),
                    9: (0, 255, 0),
                }

    def update_confidence_threshold(self, value):
        self.conf_thres = value

    def update_multicolor_classes(self, value):
        self.toggle_color_map(value)

    def update_fps_target(self, fps):
        self.fps_target = fps
        self.frame_duration = 1.0 / fps

    def stop(self):
        self.running = False

    def resume(self):
        self.running = True

    def terminate(self):
        self.stop()
        self.alive = False
        self.wait()


class DetectionProcessor(Thread):
    def __init__(self, video_path, model_path, result_queue, batch_size=4):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.running = False
        self.alive = True
        self.result_queue = result_queue
        self.batch_size = batch_size

        # Load the YOLO model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

    def run(self):
        while self.alive:
            frames = []
            print('running')
            while self.running and self.cap.isOpened():
                if not self.running:
                    break  # Exit immediately if running is set to False

                ret, frame = self.cap.read()
                if not ret or not self.running:
                    break  # Exit if reading fails or running is set to False

                frames.append(frame)
                if len(frames) == self.batch_size:
                    try:
                        results = self.model(frames)
                        for frame, result in zip(frames, results):
                            if not self.running:
                                break  # Stop processing if running is set to False
                            self.result_queue.put((frame, result))

                        frames = []
                    except Exception as e:
                        print(f"Error during detection: {e}")

            print('loop exited')
            # Cleanup if there are remaining frames and the thread is still running
            if frames:
                try:
                    results = self.model(frames)
                    for frame, result in zip(frames, results):
                        if not self.running:
                            break
                        self.result_queue.put((frame, result))
                except Exception as e:
                    print(f"Error during final batch processing: {e}")

    def is_stopped(self):
        return not self.running

    def stop(self):
        self.running = False

    def resume(self):
        self.running = True

    def terminate(self):
        self.alive = False
        self.cap.release()
        self.stop()
