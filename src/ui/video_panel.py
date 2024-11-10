import time

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSlider, QHBoxLayout, QLabel,
                             QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QMutex, QSize
from ultralytics import YOLO
import cupy as cp
import torch
import yaml
from PyQt6.QtGui import QPixmap
from src.threads import DetectionProcessor, RenderProcessor
from queue import Queue
import logging


def format_time(seconds):
    """Convert seconds to minutes:seconds format."""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"


class VideoPanel(QWidget):
    def __init__(self, video_path, model_path):
        super().__init__()

        # Set up the layout
        self.signal_connected = True
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
        self.result_queue = Queue(maxsize=10)

        # Thread Creation
        self.frame_lock = QMutex()

        # Initialize DetectionProcessor and RenderProcessor
        self.detection_processor = DetectionProcessor(video_path, model_path, self.result_queue)
        self.renderer = RenderProcessor(self.result_queue, self.detection_processor.model.names, fps_target=60)

        # Connect renderer signal to update display
        self.renderer.frame_updated.connect(self.update_displayed_frame)
        self.currently_resizing = False
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.end_resize)
        self.qt_image = None

        self.video_path = video_path
        self.model_path = model_path
        self.converting_to_pixmap = False


    def start_video(self):
        # Start processors if not running
        print('Attempting to start video')
        self.detection_processor.resume()
        self.renderer.resume()
        print(f'Detection Processor Alive: {self.detection_processor.is_alive()} Stopped: {self.detection_processor.is_stopped()}')
        if not self.detection_processor.is_alive():
            print('Starting detection processor')
            self.detection_processor.start()
        if not self.renderer.isRunning():
            print('Starting renderer')
            self.renderer.start()

    def pause_video(self):
        # Stop both processors
        print('Attempting to stop detection processor')
        self.detection_processor.stop()
        print('Attempting to stop renderer')
        self.renderer.stop()

    def update_displayed_frame(self, qt_image):
        # Store the latest image and only update if resizing is not active
        self.qt_image = qt_image
        if not self.currently_resizing:
            self.apply_image(qt_image)

    def apply_image(self, qt_image):
        min_size = QSize(100, 100)  # Set a minimum dimension for stability
        target_size = self.video_display.size().expandedTo(min_size)

        if self.converting_to_pixmap:
            return  # Skip if conversion is already in progress

        if qt_image is None or qt_image.isNull():
            print("Invalid qt_image, skipping conversion.")
            return

        self.converting_to_pixmap = True
        try:
            scaled_image = qt_image.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio)
            self.video_display.setPixmap(QPixmap.fromImage(scaled_image))
        except Exception as e:
            print(f"Error applying image: {e}")
        finally:
            self.converting_to_pixmap = False

    def stop_video(self):
        # Stop both processors
        self.detection_processor.terminate()
        self.renderer.terminate()

    def closeEvent(self, event):
        # Ensure processors stop when widget is closed
        self.stop_video()
        self.currently_resizing = True
        while self.converting_to_pixmap:
            time.sleep(0.1)
        event.accept()

    def resizeEvent(self, event):
        if not self.converting_to_pixmap:
            self.pause_video()
            self.currently_resizing = True
            self.resize_timer.start(300)  # Adjust delay as needed
            super().resizeEvent(event)
            self.start_video()

    def end_resize(self):
        # Called when resizing has stabilized
        self.currently_resizing = False
        # Now apply the latest frame
        if self.qt_image and not self.qt_image.isNull():
            self.apply_image(self.qt_image)
