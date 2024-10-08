import cv2
import yaml
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QImage
from src.video_stream import VideoStream  # Assuming the VideoStream class is already implemented

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Load video source from config file
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        if config['video']['live'] == True:
            self.video_path = int(config['video']['device'])
        else:
            self.video_path = config['video']['source']

        # Set up the UI
        self.setWindowTitle("Video Stream")
        self.video_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        # Initialize VideoStream (OpenCV)
        self.video_stream = VideoStream(self.video_path)

        # Timer to update frames every 30 ms (roughly 33 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        frame = self.video_stream.get_frame()
        if frame is not None:
            # Convert frame from BGR to RGB for PyQt
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))
        else:
            print("Error: Unable to read the video frame or end of video")
            self.timer.stop()  # Stop the timer when the video ends

    def closeEvent(self, event):
        # Release video stream resources
        self.video_stream.release()
        super().closeEvent(event)
