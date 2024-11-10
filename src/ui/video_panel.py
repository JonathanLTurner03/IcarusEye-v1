import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSlider, QHBoxLayout, QLabel,
                             QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO
from src.video_stream import VideoStream
from src.opengl_video_widget import OpenGLVideoWidget
import OpenGL.GL as gl
import cupy as cp
import torch
import yaml
from PyQt6.QtGui import QPixmap
from src.threads import DetectionProcessor, RenderProcessor
from queue import Queue


def format_time(seconds):
    """Convert seconds to minutes:seconds format."""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"


class VideoPanel(QWidget):
    def __init__(self, video_path, model_path):
        super().__init__()

        # Set up the layout
        self.layout = QVBoxLayout(self)

        # QLabel for displaying video frames
        self.video_display = QLabel(self)
        self.video_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.video_display)

        # Control buttons
        self.button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play", self)
        self.pause_button = QPushButton("Pause", self)
        self.button_layout.addWidget(self.play_button)
        self.button_layout.addWidget(self.pause_button)
        self.layout.addLayout(self.button_layout)

        # Connect button signals
        self.play_button.clicked.connect(self.start_video)
        self.pause_button.clicked.connect(self.pause_video)

        # Queues for processing
        self.result_queue = Queue()

        # Initialize DetectionProcessor and RenderProcessor
        self.detection_processor = DetectionProcessor(video_path, model_path, self.result_queue)
        self.renderer = RenderProcessor(self.result_queue, self.detection_processor.model.names, fps_target=60)

        # Connect renderer signal to update display
        self.renderer.frame_updated.connect(self.update_displayed_frame)

    def start_video(self):
        # Start processors if not running
        if not self.detection_processor.is_alive():
            self.detection_processor.start()
        if not self.renderer.isRunning():
            self.renderer.start()

    def pause_video(self):
        # Stop both processors
        self.detection_processor.stop()
        self.renderer.stop()

    def update_displayed_frame(self, qt_image):
        # Update QLabel with new frame
        if self.video_display.width() > 0 and self.video_display.height() > 0:
            pixmap = QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.video_display.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_display.setPixmap(pixmap)

    def closeEvent(self, event):
        # Ensure processors stop when widget is closed
        self.pause_video()
        event.accept()

    def resizeEvent(self, event):
        # Temporarily stop updates during resizing
        self.renderer.frame_updated.disconnect(self.update_displayed_frame)
        super().resizeEvent(event)
        # Reconnect after resizing
        self.renderer.frame_updated.connect(self.update_displayed_frame)
