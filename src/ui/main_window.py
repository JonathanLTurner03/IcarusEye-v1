import cv2
from PyQt6.QtWidgets import QMainWindow, QLabel, QPushButton
from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from PyQt6 import uic
from src.video_stream import VideoStream
from src.threads import DetectionThread, TrackingThread  # Import both threads
from src.overlays import draw_boxes
import yaml
import logging

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
log_level = config['logging']['level']

logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI from the .ui file created in Qt Designer
        uic.loadUi('src/ui/main_window.ui', self)

        # Load the configuration file
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Access the QLabel from the UI where video will be displayed
        self.video_label = self.findChild(QLabel, 'videoLabel')

        # Initialize VideoStream (OpenCV) based on config
        source = self.config['video']['source'] \
            if not self.config['video']['live'] else self.config['video']['device']

        verbose = self.config['logging']['detection_verbose']

        self.video_stream = VideoStream(source)

        # Access buttons (from the .ui file)
        self.play_button = self.findChild(QPushButton, 'startButton')
        self.pause_button = self.findChild(QPushButton, 'stopButton')

        # Initialize the DetectionThread with the YOLOv8s model
        model_path = self.config['model']['yolov8s_pretrained']
        self.detection_thread = DetectionThread(model_path, verbose)
        self.detection_thread.detection_done.connect(self.handle_detection)
        self.detection_thread.error.connect(self.handle_error)
        self.detection_thread.start()

        # Initialize the TrackingThread
        self.tracking_thread = TrackingThread()
        self.tracking_thread.tracking_done.connect(self.handle_tracking)
        self.tracking_thread.error.connect(self.handle_error)
        self.tracking_thread.start()

        # Timer to update frames every 30 ms (~33 FPS), but initially not started
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Connect buttons to methods
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)

        self.frame_counter = 0  # Counter to keep track of frames
        self.nth_frame = self.config['video']['nth_frame']  # Run detection every nth frame

        # Variables to store detection results for persistence
        self.last_boxes = []
        self.last_scores = []
        self.last_classes = []

        # Variables to store tracking results
        self.tracked_boxes = []
        self.track_ids = []

        # Default native resolution (e.g., 1080p)
        self.current_native_resolution = (1920, 1080)

        # Access detection settings
        self.confidence_threshold = self.config['detection']['confidence_threshold']
        self.omit_classes = set(self.config['detection']['omit_classes'])  # Convert to set for fast lookups

    def play_video(self):
        """Start or resume the video stream."""
        if not self.timer.isActive():
            self.timer.start(30)  # Start the timer to update frames every 30 ms

    def pause_video(self):
        """Pause the video stream."""
        if self.timer.isActive():
            self.timer.stop()  # Stop the timer to pause video updates

    def update_frame(self):
        """Update the QLabel with the next frame from the VideoStream and handle detection."""
        frame = self.video_stream.get_frame()
        if frame is not None:
            # Retrieve the frame rate from the video source
            fps = self.video_stream.get_fps()
            if fps > 0:
                self.timer.setInterval(int(1000 / fps))  # Set the timer interval based on the frame rate

            # Increment the frame counter
            self.frame_counter += 1

            # Only run detection every nth frame
            if self.frame_counter % self.nth_frame == 0:
                logging.debug("Sending frame for detection")
                self.detection_thread.send_frame.emit(frame)  # Send frame for detection

            # Draw the last detected bounding boxes (even on skipped frames)
            if hasattr(self, 'tracked_boxes') and hasattr(self, 'track_ids'):
                frame = draw_boxes(frame, self.tracked_boxes, self.track_ids)

            # Convert the frame to QImage for displaying in QLabel
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Scale the QImage to fit within the QLabel while maintaining aspect ratio
            q_image = q_image.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)

            self.video_label.setPixmap(QPixmap.fromImage(q_image))
        else:
            logging.error("Error: Unable to read the video frame or end of video")
            self.timer.stop()  # Stop the timer if the video ends



    @pyqtSlot(list, list, list, object)
    def handle_detection(self, boxes, scores, classes, frame):
        """Handle detection results and pass them to DeepSORT for tracking."""

        # Filter based on confidence and omitted classes
        self.last_boxes = []
        self.last_scores = []
        self.last_classes = []
        for (box, score, class_id) in zip(boxes, scores, classes):
            if score >= self.confidence_threshold and class_id not in self.omit_classes:
                self.last_boxes.append(box)
                self.last_scores.append(score)
                self.last_classes.append(class_id)

        # Send the detection results and frame to the tracking thread
        self.tracking_thread.send_detection.emit(self.last_boxes, self.last_scores, self.last_classes, frame)

    @pyqtSlot(list, list)
    def handle_tracking(self, tracked_boxes, track_ids):
        print(f"Tracking received: {len(tracked_boxes)} boxes, {len(track_ids)} track IDs")

        # Update instance variables
        self.tracked_boxes = tracked_boxes
        self.track_ids = track_ids


    @pyqtSlot(str)
    def handle_error(self, error_msg):
        """Handle errors emitted from the detection or tracking thread."""
        logging.error(error_msg)

    def closeEvent(self, event):
        """Release video resources and stop the detection and tracking threads when the window is closed."""
        self.video_stream.release()
        self.detection_thread.stop()
        self.tracking_thread.stop()
        super().closeEvent(event)
