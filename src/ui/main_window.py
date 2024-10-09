import cv2
from PyQt6.QtWidgets import QMainWindow, QLabel, QPushButton
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QImage
from PyQt6 import uic
from src.video_stream import VideoStream
from src.detection import YOLOv8Detection  # Import the detection class
import yaml


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
        source = self.config['video']['source'] if not self.config['video']['live'] else self.config['video']['device']
        self.video_stream = VideoStream(source)

        # Access buttons (from the .ui file)
        self.play_button = self.findChild(QPushButton, 'startButton')
        self.pause_button = self.findChild(QPushButton, 'stopButton')

        # Initialize the YOLOv8s model (using the custom model)
        self.detector = YOLOv8Detection("model/yolov8s.pt")

        # Timer to update frames every 30 ms (~33 FPS), but initially not started
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Connect buttons to methods
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)

        self.frame_counter = 0 # Counter to keep track of frames
        self.nth_frame = self.config['video']['nth_frame']  # Run detection every nth frame

        # Variables to store detection results for persistence
        self.last_boxes = []
        self.last_scores = []
        self.last_classes = []

    def play_video(self):
        """Start or resume the video stream."""
        if not self.timer.isActive():
            self.timer.start(30)  # Start the timer to update frames every 30 ms

    def pause_video(self):
        """Pause the video stream."""
        if self.timer.isActive():
            self.timer.stop()  # Stop the timer to pause video updates

    def update_frame(self):
        """Update the QLabel with the next frame from the VideoStream and run YOLO detection every nth frame."""
        frame = self.video_stream.get_frame()
        if frame is not None:
            # Increment the frame counter
            self.frame_counter += 1

            # Only run detection every nth frame
            if self.frame_counter % self.nth_frame == 0:
                # Get the native resolution of the frame
                native_height, native_width = frame.shape[:2]

                # Desired detection resolution (e.g., 680p)
                detection_width = 1210  # Width corresponding to 680p
                detection_height = 680

                # Resize the frame to the detection resolution
                resized_frame = cv2.resize(frame, (detection_width, detection_height))

                # Run detection on the resized frame
                boxes, scores, classes = self.detector.detect(resized_frame)

                # Calculate the scaling factors to stretch bounding boxes back to native resolution
                scale_x = native_width / detection_width
                scale_y = native_height / detection_height

                # Rescale bounding boxes and store them for persistence
                self.last_boxes = []
                self.last_scores = []
                self.last_classes = []
                for (box, score, class_id) in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    self.last_boxes.append([x1, y1, x2, y2])
                    self.last_scores.append(score)
                    self.last_classes.append(class_id)

            # Draw the last detected bounding boxes (even on skipped frames)
            for (box, score, class_id) in zip(self.last_boxes, self.last_scores, self.last_classes):
                x1, y1, x2, y2 = box
                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class: {int(class_id)}, Confidence: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert the frame to QImage for displaying in QLabel (in native resolution)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))
        else:
            print("Error: Unable to read the video frame or end of video")
            self.timer.stop()  # Stop the timer if the video ends

    def closeEvent(self, event):
        """Release video resources when the window is closed."""
        self.video_stream.release()
        super().closeEvent(event)
