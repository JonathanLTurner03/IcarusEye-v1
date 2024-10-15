import logging

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider, QPushButton, QCheckBox, QRadioButton, QComboBox, QFileDialog
from PyQt6.QtCore import Qt
import cv2
import os
import sys

class ConfigPanel(QWidget):
    def __init__(self, controller):
        super().__init__()

        # Used to feedback and output to the main_window
        self.controller = controller

        # Creates a virtual layout for the configuration panel
        self.__config_layout = QVBoxLayout()

        # Creates the Group Boxes for the configuration panel
        self.__input_settings = QGroupBox("Input Settings")
        self.__video_group = QGroupBox("Video Settings")
        self.__detection_group = QGroupBox("Detection Settings")

        # Defines global variables for the configuration panel
        self.__fps_label = None
        self.__confidence_label = None
        self.__omitted_classes = []

        # Initialize the video and detection settings
        self.__init_input()
        self.__init_video()
        self.__init_detection()

        # Add the config group to the main layout
        self.__config_layout.addWidget(self.__input_settings)
        self.__config_layout.addWidget(self.__video_group)
        self.__config_layout.addWidget(self.__detection_group)

        # Set the layout for the ConfigPanel
        self.setLayout(self.__config_layout)

    def __init_input(self):
        video_input_layout = QVBoxLayout()

        # Radio buttons to switch between device and file input
        device_radio = QRadioButton("Device Input")
        file_radio = QRadioButton("File Input")
        device_radio.setChecked(True)  # Default to device input

        device_radio.toggled.connect(self.__toggle_input_type)

        # Dropdown for available input devices
        self.__device_dropdown = QComboBox()
        self.__populate_device_dropdown()

        # File input button
        self.__file_button = QPushButton("Select Video File")
        self.__file_button.clicked.connect(self.__select_video_file)
        self.__file_button.setEnabled(False)  # Initially disabled

        # Add widgets to the video input layout
        video_input_layout.addWidget(device_radio)
        video_input_layout.addWidget(self.__device_dropdown)
        video_input_layout.addWidget(file_radio)
        video_input_layout.addWidget(self.__file_button)

        # Set the layout for the input settings group
        self.__input_settings.setLayout(video_input_layout)

    def __init_video(self):
        # Frame Rate Slider
        self.__fps_label = QLabel(f"Frame Rate (FPS): {self.controller.fps}, Native: {self.controller.native_fps}")
        fps_slider = QSlider(Qt.Orientation.Horizontal)
        fps_slider.setRange(1, 60)
        fps_slider.setValue(self.controller.fps)

        # Connect the slider's valueChanged signal to the update method
        fps_slider.valueChanged.connect(self.update_fps)

        # Add to layout
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.__fps_label)
        video_layout.addWidget(fps_slider)

        # Set the layout for the video group
        self.__video_group.setLayout(video_layout)

    def __init_detection(self):
        # Confidence Threshold Slider
        self.__confidence_label = QLabel(f"Confidence Threshold: {self.controller.confidence}")
        confidence_slider = QSlider(Qt.Orientation.Horizontal)
        confidence_slider.setRange(0, 100)
        confidence_slider.setValue(self.controller.confidence)

        # Connect the slider's valueChanged signal to the update method
        confidence_slider.valueChanged.connect(self.update_confidence)

        # Add to layout
        detection_layout = QVBoxLayout()
        detection_layout.addWidget(self.__confidence_label)
        detection_layout.addWidget(confidence_slider)

        # Set the layout for the detection group
        self.__detection_group.setLayout(detection_layout)

    def __toggle_input_type(self):
        """Toggle between device input and file input."""
        if self.__device_dropdown.isEnabled():
            self.__device_dropdown.setEnabled(False)
            self.__file_button.setEnabled(True)
        else:
            self.__device_dropdown.setEnabled(True)
            self.__file_button.setEnabled(False)

    def __select_video_file(self):
        """Open a file dialog to select a video file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.controller.set_video_file(file_name)

    # TODO: Abstract this method to the main window.
    def __populate_device_dropdown(self):
        """Populate the dropdown with available video input devices."""
        self.__device_dropdown.clear()
        index = 0

        while True:
            # Redirect stderr to suppress camera indexing errors
            sys.stderr = open(os.devnull, 'w')
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            sys.stderr = sys.__stderr__  # Restore stderr

            if not cap.read()[0]:
                break
            self.__device_dropdown.addItem(f"Device {index}")
            cap.release()
            index += 1

    # Updates the fps slider value and label
    def update_fps(self, value):
        """Update the label and perform actions when the slider value changes."""
        self.__fps_label.setText(f"Frame Rate (FPS): {value}, Native: {self.controller.native_fps}")
        self.controller.set_fps(value)

    # Updates the confidence slider value and label
    def update_confidence(self, value):
        """Update the label and perform actions when the slider value changes."""
        self.__confidence_label.setText(f"Confidence Threshold: {value}")
        self.controller.set_confidence(value)