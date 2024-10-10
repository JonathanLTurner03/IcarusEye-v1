from PyQt6.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
from src.detection import YOLOv8Detection
from src.tracking import Tracker
import traceback
import logging
import cv2

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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

# New TrackingWorker and TrackingThread

class TrackingWorker(QObject):
    tracking_done = pyqtSignal(list, list)  # boxes and track_ids
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.tracker = Tracker()
        self._running = True

    @pyqtSlot(list, list, list, object)
    def process_tracking(self, boxes, scores, classes, frame):
        try:
            # Get the original frame dimensions
            original_height, original_width = frame.shape[:2]

            # Run DeepSORT on the original frame and boxes
            tracked_objects = self.tracker.update_tracks(boxes, scores, classes, frame)

            # Rescale bounding boxes back to the original frame size for UI display
            final_boxes = []
            track_ids = []
            for obj in tracked_objects:
                box = obj['bbox']
                x1, y1, x2, y2 = box

                # Ensure the bounding box coordinates are integers
                x1 = max(0, min(int(x1), original_width))
                y1 = max(0, min(int(y1), original_height))
                x2 = max(0, min(int(x2), original_width))
                y2 = max(0, min(int(y2), original_height))

                final_boxes.append([x1, y1, x2, y2])
                track_ids.append(obj['track_id'])

            # Emit the final boxes and track IDs to be drawn on the original frame
            self.tracking_done.emit(final_boxes, track_ids)

        except Exception as e:
            error_msg = f"Tracking Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def stop(self):
        self._running = False



    def stop(self):
        self._running = False

class TrackingThread(QThread):
    # Signal to send detection results to the worker
    send_detection = pyqtSignal(list, list, list, object)  # boxes, scores, classes

    # Signals to receive tracking results or errors
    tracking_done = pyqtSignal(list, list)  # Tracked boxes and IDs
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.worker = TrackingWorker()

    def run(self):
        # Move the worker to this thread
        self.worker.moveToThread(self)

        # Connect signals and slots
        self.send_detection.connect(self.worker.process_tracking)
        self.worker.tracking_done.connect(self.tracking_done)
        self.worker.error.connect(self.error)

        # Start the event loop
        self.exec()

    def stop(self):
        self.worker.stop()
        self.quit()
        self.wait()
