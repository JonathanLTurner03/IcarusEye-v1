from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider, QPushButton, QCheckBox
from PyQt6.QtCore import Qt


class ConfigPanel(QWidget):
    def __init__(self, controller):
        super().__init__()

        # Used to feedback and output to the main_window
        self.controller = controller

        # Creates a virtual layout for the configuration panel
        self.__config_layout = QVBoxLayout()

        # Creates the Group Boxes for the configuration panel
        self.__video_group = QGroupBox("Video Settings")
        self.__detection_group = QGroupBox("Detection Settings")

        # Defines global variables for the configuration panel
        self.__fps_label = None
        self.__confidence_label = None

        # Initialize the video and detection settings
        self.__init_video()
        self.__init_detection()

        # Add the config group to the main layout
        self.__config_layout.addWidget(self.__video_group)
        self.__config_layout.addWidget(self.__detection_group)

        # Set the layout for the ConfigPanel
        self.setLayout(self.__config_layout)

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