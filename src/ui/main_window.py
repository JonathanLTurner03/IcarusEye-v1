import cv2
from PyQt6.QtWidgets import QMainWindow, QLabel, QPushButton
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QImage
from PyQt6 import uic
from src.video_stream import VideoStream
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

        # Access buttons (from the .ui file)
        self.play_button = self.findChild(QPushButton, 'startButton')
        self.pause_button = self.findChild(QPushButton, 'stopButton')
        self.video_stream = VideoStream(source)

        # Timer to update frames every n ms, but initially not started
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Connect buttons to methods
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)

    def play_video(self):
        """Start or resume the video stream."""
        if not self.timer.isActive():
            self.timer.start(17)  # Start the timer to update frames every 17 ms

    def pause_video(self):
        """Pause the video stream and reset QLabel."""
        if self.timer.isActive():
            self.timer.stop()  # Stop the timer to pause video updates

            # Clear the QLabel and set it to show "Video Paused" or any placeholder text/image
            self.video_label.clear()
            self.video_label.setText("Video Paused")  # You can replace this with a placeholder image if desired
            self.video_label.setStyleSheet("QLabel { background-color : black; color : white; }")  # Optional styling

    def update_frame(self):
        """Update the QLabel with the next frame from the VideoStream."""
        frame = self.video_stream.get_frame()
        if frame is not None:
            # Convert the frame to QImage for displaying in QLabel
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
