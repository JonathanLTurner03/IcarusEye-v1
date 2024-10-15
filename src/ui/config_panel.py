from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider, QPushButton, QCheckBox
from PyQt6.QtCore import Qt


class ConfigPanel(QWidget):
    def __init__(self, controller):
        super().__init__()

        # Used to feedback and output to the main_window
        self.controller = controller

        # Creates a virtual layout for the configuration panel
        self.layout = QVBoxLayout()

        # Creates a group box to hold the configuration options
        self.config_group = QGroupBox("Configuration")
        self.config_layout = QVBoxLayout()

        # Frame Rate Slider
        self.fps_label = QLabel(f"Frame Rate (FPS): {self.controller.fps}, Native: {self.controller.native_fps}")
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(1, 60)
        self.fps_slider.setValue(self.controller.fps)

        # Connect the slider's valueChanged signal to the update method
        self.fps_slider.valueChanged.connect(self.update_fps)

        # Confidence Threshold Slider
        self.confidence_label = QLabel(f"Confidence Threshold: {self.controller.confidence}")
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(self.controller.confidence)

        # Connect the slider's valueChanged signal to the update method
        self.confidence_slider.valueChanged.connect(self.update_confidence)

        self.config_layout.addWidget(self.fps_label)
        self.config_layout.addWidget(self.fps_slider)
        self.config_layout.addWidget(self.confidence_label)
        self.config_layout.addWidget(self.confidence_slider)

        # Set the layout for the config group
        self.config_group.setLayout(self.config_layout)

        # Add the config group to the main layout
        self.layout.addWidget(self.config_group)

        # Set the layout for the ConfigPanel
        self.setLayout(self.layout)


    # Updates the fps slider value and label
    def update_fps(self, value):
        """Update the label and perform actions when the slider value changes."""
        self.fps_label.setText(f"Frame Rate (FPS): {value}, Native: {self.controller.native_fps}")
        self.controller.set_fps(value)

    # Updates the confidence slider value and label
    def update_confidence(self, value):
        """Update the label and perform actions when the slider value changes."""
        self.confidence_label.setText(f"Confidence Threshold: {value}")
        self.controller.set_confidence(value)