from PyQt6.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
import torch
from src.detection import YOLOv8Detection
import traceback
import logging
import cv2
import os
import sys

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


class DeviceScanner(QObject):
    devices_scanned = pyqtSignal(list)

    def run(self):
        """Scan for available video input devices."""
        devices = []
        index = 0

        while True:
            # Redirect stderr to suppress camera indexing errors
            sys.stderr = open(os.devnull, 'w')
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            sys.stderr = sys.__stderr__  # Restore stderr

            if not cap.read()[0]:
                break
            devices.append(f"Device {index}")
            cap.release()
            index += 1

        self.devices_scanned.emit(devices)