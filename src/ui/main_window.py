import cv2
from PyQt6.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from PyQt6 import uic
from src.video_stream import VideoStream
from src.opengl_video_widget import OpenGLVideoWidget  # Import your OpenGL widget
from src.threads import DetectionThread  # Import the DetectionThread
from src.overlays import draw_boxes
import yaml
import logging


with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
log_level = config['logging']['level']

logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


# TODO add documentation and comments
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

        # Initialize the DetectionThread with the YOLOv8s model
        model_path = self.config['model']['yolov8s']
        self.detection_thread = DetectionThread(model_path, verbose)
        self.detection_thread.detection_done.connect(self.handle_detection)
        self.detection_thread.error.connect(self.handle_error)
        self.detection_thread.start()

        # Access buttons (from the .ui file)
        self.play_button = self.findChild(QPushButton, 'startButton')
        self.pause_button = self.findChild(QPushButton, 'stopButton')

        # Create an instance of the OpenGL widget and replace the QLabel
        self.video_widget = OpenGLVideoWidget(self)
        self.videoLayout = self.findChild(QVBoxLayout, 'videoLayout')  # Assuming you have a layout
        self.videoLayout.addWidget(self.video_widget)  # Add OpenGL widget to the layout

        # Remove the QLabel if necessary
        self.videoLabel = self.findChild(QLabel, 'videoLabel')
        self.videoLabel.setVisible(False)  # Optionally hide the original QLabel

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

        # Default native resolution (e.g., 720p)
        self.current_native_resolution = (1920, 1080)

        # Access detection settings
        self.confidence_threshold = self.config['detection']['confidence_threshold']
        self.max_labels = self.config['video']['max_labels']
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
        """Update the OpenGL widget with the next frame from the VideoStream."""
        frame = self.video_stream.get_frame()
        if frame is not None:
            self.frame_counter += 1

            # Upload the captured frame to OpenGL
            self.video_widget.upload_frame_to_opengl(frame)

            # Trigger a repaint
            self.video_widget.update()  # This will call paintGL
        else:
            print("Error: Unable to read the video frame or end of video")
            self.timer.stop()  # Stop the timer if the video ends

    @pyqtSlot(list, list, list)
    def handle_detection(self, boxes, scores, classes):
        """Handle detection results emitted from the detection thread."""
        if hasattr(self, 'current_native_resolution'):
            native_width, native_height = self.current_native_resolution
            detection_width = 640  # The width after resizing in the detection model
            detection_height = 640  # The height after resizing in the detection model

            # Calculate the scaling factors to stretch bounding boxes back to native resolution
            scale_x = native_width / detection_width
            scale_y = native_height / detection_height

            # Rescale bounding boxes and filter based on confidence and omitted classes
            self.last_boxes = []
            self.last_scores = []
            self.last_classes = []
            for (box, score, class_id) in zip(boxes, scores, classes):
                # Check if the confidence is above the threshold and the class is not in omitted classes
                if score >= self.confidence_threshold and class_id not in self.omit_classes:
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    self.last_boxes.append([x1, y1, x2, y2])
                    self.last_scores.append(score)
                    self.last_classes.append(class_id)

            # Update the OpenGL widget with the new detection results
            self.video_widget.bounding_boxes = self.last_boxes  # Update bounding boxes for OpenGL widget


    @pyqtSlot(str)
    def handle_error(self, error_msg):
        """Handle errors emitted from the detection thread."""
        logging.error(error_msg)

    def closeEvent(self, event):
        """Release video resources and stop the detection thread when the window is closed."""
        self.video_stream.release()
        self.detection_thread.stop()
        super().closeEvent(event)
