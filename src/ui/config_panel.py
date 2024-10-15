import logging

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider, QPushButton, QCheckBox, QRadioButton,
                              QComboBox, QFileDialog, QSpacerItem, QSizePolicy, QHBoxLayout)
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
        self.__fps_slider = None

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
        fps_slider.valueChanged.connect(self.update_fps)

        # Common FPS Buttons
        fps_buttons_layout = QHBoxLayout()
        for fps in [24, 30, 60]:
            button = QPushButton(f"{fps} FPS")
            button.clicked.connect(lambda _, f=fps: self.__set_fps(f, fps_slider))
            fps_buttons_layout.addWidget(button)

        # Resolution Settings
        self.__resolution_label = QLabel("Resolution:")
        self.__resolution_dropdown = QComboBox()
        resolutions = ["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "1.75x", "2x"]
        self.__resolution_dropdown.addItems(resolutions)
        self.__resolution_dropdown.currentIndexChanged.connect(self.update_resolution)

        # Add to layout
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.__fps_label)
        video_layout.addWidget(fps_slider)
        video_layout.addLayout(fps_buttons_layout)
        video_layout.addWidget(self.__resolution_label)
        video_layout.addWidget(self.__resolution_dropdown)

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

        # Omit Classes Section
        self.__omit_classes_checkbox = QCheckBox("Omit Classes")
        self.__omit_classes_checkbox.stateChanged.connect(self.__toggle_omit_classes)

        self.__classes_dropdown = QComboBox()
        self.__classes_dropdown.addItems(self.controller.get_available_classes())
        self.__classes_dropdown.setEnabled(False)

        self.__add_class_button = QPushButton("Add Class")
        self.__add_class_button.clicked.connect(self.__add_class_to_omit)
        self.__add_class_button.setEnabled(False)

        self.__remove_class_button = QPushButton("Remove Class")
        self.__remove_class_button.clicked.connect(self.__remove_class_from_omit)
        self.__remove_class_button.setEnabled(False)

        self.__omitted_classes_label = QLabel("Omitted Classes: None")

        # Add to layout
        detection_layout = QVBoxLayout()
        detection_layout.addWidget(self.__confidence_label)
        detection_layout.addWidget(confidence_slider)
        detection_layout.addWidget(self.__omit_classes_checkbox)
        detection_layout.addWidget(self.__classes_dropdown)
        detection_layout.addWidget(self.__add_class_button)
        detection_layout.addWidget(self.__remove_class_button)
        detection_layout.addWidget(self.__omitted_classes_label)

        # Set the layout for the detection group
        self.__detection_group.setLayout(detection_layout)

    def update_resolution(self, index):
        """Update the resolution based on the selected index."""
        resolution = self.__resolution_dropdown.itemText(index)
        # Implement the logic to update the resolution in the controller
        print(f"Selected resolution: {resolution}")

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

    def __toggle_omit_classes(self, state):
        """Enable or disable the omit classes section."""
        enabled = state
        self.__classes_dropdown.setEnabled(enabled)
        self.__add_class_button.setEnabled(enabled)
        self.__remove_class_button.setEnabled(enabled)

    def __add_class_to_omit(self):
        """Add the selected class to the omitted classes list."""
        selected_class = self.__classes_dropdown.currentText()
        if selected_class and selected_class not in self.__omitted_classes:
            self.__omitted_classes.append(selected_class)
            self.__update_omitted_classes_label()

    def __remove_class_from_omit(self):
        """Remove the selected class from the omitted classes list."""
        selected_class = self.__classes_dropdown.currentText()
        if selected_class in self.__omitted_classes:
            self.__omitted_classes.remove(selected_class)
            self.__update_omitted_classes_label()

    def __update_omitted_classes_label(self):
        """Update the label displaying the omitted classes."""
        if self.__omitted_classes:
            omitted_classes_text = ", ".join(self.__omitted_classes)
        else:
            omitted_classes_text = "None"
        self.__omitted_classes_label.setText(f"Omitted Classes: {omitted_classes_text}")

    def __set_fps(self, value, slider):
        """Set the FPS value and update the slider."""
        self.controller.set_fps(value)
        slider.setValue(value)

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