import time

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSlider, QHBoxLayout, QLabel,
                             QSizePolicy, QMessageBox, QDialog, QSpinBox, QLineEdit, QRadioButton)
from PyQt6.QtCore import Qt, QTimer, QMutex, QSize
from ultralytics import YOLO
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

        self.previous_time = None

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
        self.detection_processor.resume()
        self.renderer.resume()
        if not self.detection_processor.is_alive():
            self.detection_processor.start()
        if not self.renderer.isRunning():
            self.renderer.start()

    def pause_video(self):
        if self.detection_processor is None or self.detection_processor is None:
            return

        # Stop both processors
        self.detection_processor.stop()
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

        # Compute and update FPS
        current_time = time.time()
        if self.previous_time is not None and self.previous_time != current_time:
            fps = 1.0 / (current_time - self.previous_time)
            self.fps_label.setText(f"FPS: {fps:.2f}")
        else:
            self.fps_label.setText("FPS: 0.0")
        self.previous_time = current_time

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

    def update_nth_frame(self, value):
        """Update the nth frame value."""
        self.detection_processor.update_nth_frame(value)


    def setup_videocapture(self, video_device, fps_target=60, codec=None, resolution=(1280, 720)):
        """Set up video capture using a video device or file."""
        video_stream = cv2.VideoCapture(video_device)

        # Configure video stream properties if it's a number (camera device)
        if isinstance(video_device, int) and codec is not None:
            # Set codec
            video_stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
            # Set resolution
            width, height = resolution
            video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Set FPS
            video_stream.set(cv2.CAP_PROP_FPS, fps_target)

        self.detection_processor = DetectionProcessor(video_stream, self.model_path, self.result_queue)
        self.renderer = RenderProcessor(self.result_queue, self.detection_processor.model.names, fps_target=fps_target)

        # Connect renderer signal to update display
        self.renderer.frame_updated.connect(self.update_displayed_frame)

    def update_max_boxes(self, value):
        self.renderer.update_max_boxes(value)

    def prompt_video_settings(self, video_device):
        """Display a dialog to customize FPS, codec, and resolution for a camera device."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Video Settings")
        layout = QVBoxLayout(dialog)

        # Mode selection (Manual / Automatic)
        mode_layout = QHBoxLayout()
        manual_radio = QRadioButton("Manual", dialog)
        auto_radio = QRadioButton("Automatic", dialog)
        manual_radio.setChecked(True)  # Default to Manual
        mode_layout.addWidget(manual_radio)
        mode_layout.addWidget(auto_radio)
        layout.addLayout(mode_layout)

        # FPS input
        fps_layout = QHBoxLayout()
        fps_label = QLabel("FPS:", dialog)
        fps_input = QSpinBox(dialog)
        fps_input.setRange(1, 120)
        fps_input.setValue(30)  # Default FPS
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(fps_input)
        layout.addLayout(fps_layout)

        # Codec input
        codec_layout = QHBoxLayout()
        codec_label = QLabel("Codec:", dialog)
        codec_input = QLineEdit(dialog)
        codec_input.setPlaceholderText("e.g., MJPG")
        codec_input.setText("MJPG")  # Default codec
        codec_layout.addWidget(codec_label)
        codec_layout.addWidget(codec_input)
        layout.addLayout(codec_layout)

        # Resolution input
        resolution_layout = QHBoxLayout()
        resolution_label = QLabel("Resolution (WxH):", dialog)
        resolution_input = QLineEdit(dialog)
        resolution_input.setPlaceholderText("e.g., 1280x720")
        resolution_input.setText("1280x720")  # Default resolution
        resolution_layout.addWidget(resolution_label)
        resolution_layout.addWidget(resolution_input)
        layout.addLayout(resolution_layout)

        # Submit and Cancel buttons
        button_layout = QHBoxLayout()
        submit_button = QPushButton("Submit", dialog)
        cancel_button = QPushButton("Cancel", dialog)
        button_layout.addWidget(submit_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Disable inputs when Automatic is selected
        def toggle_inputs():
            is_manual = manual_radio.isChecked()
            fps_input.setVisible(is_manual)
            codec_input.setVisible(is_manual)
            resolution_input.setVisible(is_manual)

        manual_radio.toggled.connect(toggle_inputs)
        auto_radio.toggled.connect(toggle_inputs)

        # Handle dialog result
        def accept_settings():
            if auto_radio.isChecked():
                # Automatic mode: Pass video_device with None for other parameters
                self.setup_videocapture(video_device, fps_target=None, codec=None, resolution=None)
            else:
                # Manual mode: Validate and apply settings
                resolution = resolution_input.text()
                try:
                    width, height = map(int, resolution.split('x'))
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Resolution must be in the format WxH, e.g., 1920x1080.")
                    return

                # Apply settings
                fps = fps_input.value()
                codec = codec_input.text()

                # Set up video capture with manual settings
                self.setup_videocapture(video_device, fps_target=fps, codec=codec, resolution=(width, height))

            dialog.accept()

        submit_button.clicked.connect(accept_settings)
        cancel_button.clicked.connect(dialog.reject)

        dialog.exec()

    def update_detection(self, value):
        self.detection_processor.update_tracking(value)
        self.renderer.update_tracking(value)

    def update_omitted_classes(self, classes):
        self.renderer.update_omitted_classes(classes)
