import time

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSlider, QHBoxLayout, QLabel,
                             QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QMutex, QSize
from ultralytics import YOLO
import cupy as cp
import torch
import yaml
from PyQt6.QtGui import QPixmap, QImage
from src.threads import DetectionProcessor, RenderProcessor
from queue import Queue
import logging
import cv2


def format_time(seconds):
    """Convert seconds to minutes:seconds format."""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"


class VideoPanel(QWidget):
    def __init__(self, model_path):
        super().__init__()

        # Set up the layout
        self.signal_connected = True
        self.layout = QVBoxLayout(self)

        # QLabel for displaying video frames
        self.fps_label = QLabel("FPS: 0.0", self)
        self.fps_label.setEnabled(False)
        self.layout.addWidget(self.fps_label)
        self.video_display = QLabel(self)
        self.video_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_display)

        # Control button
        self.button_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play", self)
        self.stop_button = QPushButton("Stop", self)
        self.button_layout.addWidget(self.play_pause_button)
        self.button_layout.addWidget(self.stop_button)
        self.layout.addLayout(self.button_layout)

        # Connect button signal
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.stop_button.clicked.connect(self.stop_video)

        # Queues for processing
        self.result_queue = Queue(maxsize=100)

        # Thread Creation
        self.frame_lock = QMutex()

        # Initialize DetectionProcessor and RenderProcessor
        self.detection_processor = None
        self.renderer = None

        # Resizing thingies
        self.currently_resizing = False
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.end_resize)
        self.qt_image = None

        self.video_path = None
        self.model_path = model_path
        self.converting_to_pixmap = False

    def toggle_play_pause(self):
        if self.detection_processor is None or self.detection_processor is None:
            return

        if self.detection_processor.is_stopped():
            self.start_video()
            self.play_pause_button.setText("Pause")
        else:
            self.pause_video()
            self.play_pause_button.setText("Play")

    def start_video(self):
        if self.detection_processor is None or self.detection_processor is None:
            return

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
        if self.detection_processor is None or self.detection_processor is None:
            return

        # Stop both processors
        print('Attempting to stop detection processor')
        self.detection_processor.stop()
        print('Attempting to stop renderer')
        self.renderer.stop()

    def update_displayed_frame(self, frame: np.ndarray):
        # Convert the numpy array to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Set the QPixmap from QImage
        pixmap = QPixmap.fromImage(qt_image)
        self.video_display.setPixmap(pixmap)

    def update_fps_display(self, fps):
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def stop_video(self):
        if self.detection_processor is None or self.detection_processor is None:
            return

        self.pause_video()

        # Stop both processors
        self.detection_processor.terminate()
        self.renderer.terminate()

        # Clear the video display
        self.video_display.clear()
        self.detection_processor = None
        self.renderer = None

    def resizeEvent(self, event):
        if self.detection_processor is None or self.detection_processor is None:
            return

        was_paused = self.detection_processor.is_stopped()
        if not self.converting_to_pixmap:
            if not was_paused:
                self.pause_video()
            self.currently_resizing = True
            if not was_paused:
                self.resize_timer.start(300)  # Adjust delay as needed
            else:
                self.resize_timer.start(0)
            super().resizeEvent(event)
            if not was_paused:
                self.start_video()

    def end_resize(self):
        # Called when resizing has stabilized
        self.currently_resizing = False
        # Now apply the latest frame
        if self.qt_image and not self.qt_image.isNull():
            self.apply_image(self.qt_image)


    def update_confidence_threshold(self, value):
        """Update the confidence threshold for the detection model."""
        self.renderer.update_confidence_threshold(value)

    def update_colormap(self, value):
        """Update the colormap value."""
        self.renderer.update_multicolor_classes(value)


    def setup_videocapture(self, video_path, fps_target=60):
        self.detection_processor = DetectionProcessor(video_path, self.model_path, self.result_queue)
        self.renderer = RenderProcessor(self.result_queue, self.detection_processor.model.names, fps_target=fps_target)

        # Connect renderer signal to update display
        self.renderer.frame_updated.connect(self.update_displayed_frame)
        self.renderer.fps_updated.connect(self.update_fps_display)
