from PyQt6.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
import subprocess
from src.detection import YOLOv8Detection
import traceback
import logging
import cv2
import re

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# TODO add documentation and comments
class DetectionWorker(QObject):
    detection_done = pyqtSignal(list, list, list, object)  # boxes, scores, classes
    error = pyqtSignal(str)

    def __init__(self, model_path, verbose=False):
        super().__init__()
        self.detector = YOLOv8Detection(model_path)
        self._running = True
        self.verbose = verbose

    @pyqtSlot(object)
    def process_frame(self, frame):
        try:
            logging.debug("Received frame for detection")
            boxes, scores, classes = self.detector.detect(frame, self.verbose)

            # Ensure boxes, scores, and classes are lists before emitting
            boxes = boxes.tolist() if not isinstance(boxes, list) else boxes
            scores = scores.tolist() if not isinstance(scores, list) else scores
            classes = classes.tolist() if not isinstance(classes, list) else classes

            logging.debug(f"Emitting detection results: boxes={boxes}, scores={scores}, classes={classes}")
            self.detection_done.emit(boxes, scores, classes, frame)
        except Exception as e:
            error_msg = f"Detection Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def stop(self):
        self._running = False


# TODO add documentation and comments
class DetectionThread(QThread):
    # Signal to send frames to the worker
    send_frame = pyqtSignal(object)

    # Signals to receive detection results or errors
    detection_done = pyqtSignal(list, list, list, object)
    error = pyqtSignal(str)

    def __init__(self, model_path, verbose=False):
        super().__init__()
        self.worker = DetectionWorker(model_path, verbose)

    def run(self):
        # Move the worker to this thread
        self.worker.moveToThread(self)

        # Connect signals and slots
        self.send_frame.connect(self.worker.process_frame)
        self.worker.detection_done.connect(self.detection_done)
        self.worker.error.connect(self.error)

        # Start the event loop
        self.exec()

    def stop(self):
        self.worker.stop()
        self.quit()
        self.wait()

def is_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

def list_ffmpeg_devices():
    if not is_ffmpeg_installed():
        print("FFmpeg is not installed. Please install FFmpeg to use this function.")
        return []

    devices = []
    try:
        result = subprocess.run(
            ['ffmpeg', '-f', 'dshow', '-list_devices', 'true', '-i', 'dummy'],
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stderr
        device_pattern = re.compile(r'\[dshow @ [^]]+] "([^"]+)" \(video\)')
        devices = device_pattern.findall(output)
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


def list_ffmpeg_device_details(device_name):
    try:
        command = ['ffmpeg', '-f', 'dshow', '-list_options', 'true', '-i', f'video={device_name}']
        result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
        output = result.stderr

        # Pattern to match vcodec details with resolution and framerate
        detail_pattern = re.compile(
            r'vcodec=([^ ]+) +min s=(\d+x\d+) fps=(\d+(\.\d+)?) +max s=(\d+x\d+) fps=(\d+(\.\d+)?)'
        )

        codecs_info = {}

        lines = output.splitlines()
        for line in lines:
            # Check if the line contains a codec (vcodec) and its details
            detail_match = detail_pattern.search(line)
            if detail_match:
                codec, min_res, min_fps, _, max_res, max_fps, _ = detail_match.groups()
                if codec not in codecs_info:
                    codecs_info[codec] = {
                        'codec': codec,
                        'resolutions': []
                    }

                resolution_entry = {
                    'min_resolution': min_res,
                    'max_resolution': max_res,
                    'min_fps': float(min_fps),
                    'max_fps': float(max_fps)
                }

                # Add the resolution details under the matched codec
                if resolution_entry not in codecs_info[codec]['resolutions']:
                    codecs_info[codec]['resolutions'].append(resolution_entry)

        return list(codecs_info.values())

    except Exception as e:
        print(f"Error getting device details with FFmpeg: {e}")
        return []


class DeviceScanner(QObject):
    devices_scanned = pyqtSignal(list)

    def run(self):
        """Scan for available video input devices and their details."""
        device_info = []
        if is_ffmpeg_installed():
            ffmpeg_devices = list_ffmpeg_devices()
            opencv_devices = list_opencv_devices(max_devices=len(ffmpeg_devices))
            for index, device_name in zip(opencv_devices, ffmpeg_devices):
                details = list_ffmpeg_device_details(device_name)
                device_info.append({
                    'index': index,
                    'name': device_name,
                    'details': details
                })
        else:
            for index in list_opencv_devices():
                device_info.append({
                    'index': index,
                    'name': f"Device {index}",
                    'details': []
                })
        self.devices_scanned.emit(device_info)