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
        self.max_boxes = 100
        self.toggle_color_map(False)

    def run(self):
        while self.alive:
            while self.running:
                start_time = time.time()

                try:
                    frame, results = self.result_queue.get(timeout=1)
                except:
                    print("No frame in queue; continuing...")
                    continue

                try:
                    boxes = results.boxes
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    xyxy_boxes = boxes.xyxy.cpu().numpy().astype(int)

                    # Access tracking IDs if available
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        tracking_ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        tracking_ids = [None] * len(boxes)

                    # Combine all information
                    for i in range(len(boxes)):
                        if confidences[i] >= self.conf_thres:
                            x1, y1, x2, y2 = xyxy_boxes[i]
                            conf = confidences[i]
                            cls = class_ids[i]
                            label = f"{self.model_names[cls]}: {conf:.2f}"

                            # Include tracking ID if available
                            if tracking_ids[i] is not None:
                                label = f"ID {tracking_ids[i]} {label}"

                            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_map[cls], 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, self.color_map[cls], 2)

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

    def update_max_boxes(self, value):
        self.max_boxes = value

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
        self.cap = video_path
        self.running = False
        self.alive = True
        self.result_queue = result_queue
        self.batch_size = batch_size
        self.nth_frame = 1
        self.conf_thres = 0.5

        # Load the YOLO model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

        # Add tracking attribute
        self.use_tracking = True  # Set tracking to be on by default

        # Set tracker configuration path
        self.tracker_config_path = 'models/bytetrack.yaml'  # Use default tracker config

    def run(self):
        while self.alive:
            frames = []

            while self.running and self.cap.isOpened():
                if not self.running:
                    break  # Exit immediately if running is set to False

                ret, frame = self.cap.read()
                if not ret or not self.running:
                    break  # Exit if reading fails or running is set to False

                frames.append(frame)
                if len(frames) == self.batch_size:
                    try:
                        if self.use_tracking:
                            # Use model.track() when tracking is enabled
                            results = self.model.track(source=frames,
                                                       conf=self.conf_thres,
                                                       tracker=self.tracker_config_path,
                                                       verbose=False)
                        else:
                            # Use model.predict() when tracking is disabled
                            results = self.model.predict(source=frames,
                                                         conf=self.conf_thres,
                                                         verbose=False)

                        for frame, result in zip(frames, results):
                            if not self.running:
                                break  # Stop processing if running is set to False
                            self.result_queue.put((frame, result))

                        frames = []
                    except Exception as e:
                        print(f"Error during detection: {e}")

            # Cleanup if there are remaining frames and the thread is still running
            if frames:
                try:
                    if self.use_tracking:
                        results = self.model.track(source=frames,
                                                   conf=self.conf_thres,
                                                   tracker=self.tracker_config_path,
                                                   verbose=False)
                    else:
                        results = self.model.predict(source=frames,
                                                     conf=self.conf_thres,
                                                     verbose=False)

                    for frame, result in zip(frames, results):
                        if not self.running:
                            break
                        self.result_queue.put((frame, result))
                except Exception as e:
                    print(f"Error during final batch processing: {e}")

    # Methods to update tracking
    def update_tracking(self, value):
        self.use_tracking = value

    def enable_tracking(self):
        self.use_tracking = True

    def disable_tracking(self):
        self.use_tracking = False

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