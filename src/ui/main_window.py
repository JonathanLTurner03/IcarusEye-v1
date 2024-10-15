import time
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from src.ui.config_panel import ConfigPanel
from src.ui.video_panel import VideoPanel
import cv2
import os
import sys
import yaml


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the configuration file
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Setup properties
        self.fps = 30
        self.native_fps = 30
        self.confidence = 50
        self.__res_multiplier = 1.0
        self.__nth_frame = 5
        self.__bbox_max = 100

        # Get the available classes
        self.__class_details = self.config['class_details']
        self.__multi_color_classes = False
        self.__omit_classes = []

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set up the layout for the central widget
        self.layout = QHBoxLayout(self.central_widget)

        # Create and add the ConfigPanel to the layout
        self.config_panel = ConfigPanel(self)
        self.video_panel = VideoPanel(self)

        self.layout.addWidget(self.video_panel)
        self.layout.addWidget(self.config_panel)

        # Set the stretch factors for the layout
        self.layout.setStretch(0, 7)  # VideoPanel takes 70% of the space
        self.layout.setStretch(1, 3)  # ConfigPanel takes 30% of the space

        self.resize(1600, 900)  # Width: 800, Height: 600


    # TODO: Implement the following methods

    def set_fps(self, value):
        """Set the FPS value."""
        # TODO: Implement the update to the video player
        self.fps = value

    def set_confidence(self, value):
        """Set the confidence threshold value."""
        # TODO: Implement the update to the detection worker
        self.confidence = value

    # Helper functions

    # UI Value Setters and Getters #

    # Sets the resolution multiplier
    def set_video_file(self, file_path):
        """Set the video file path."""
        self.video_file = file_path
        print(f'Video file: {file_path}')

    def set_resolution_multiplier(self, value):
        """Set the resolution multiplier value."""
        self.__res_multiplier = value
        print(f'Resolution multiplier: {value}')

    # Gets the list of available classes
    def get_available_classes(self):
        """Get the list of available classes."""
        if not self.__class_details:
            return []

        if self.__multi_color_classes:
            return [f"{details['class']}: ({details['name']})" for details in self.__class_details.values()]
        return [details['class'] for details in self.__class_details.values()]

    def set_multi_color_classes(self, value):
        """Set the multi-color classes value."""
        self.__multi_color_classes = value
        print(f"Multi-color classes: {value}")

    def update_omitted_classes(self, classes):
        """Update the omitted classes."""
        self.__omit_classes = classes

    def set_nth_frame(self, value):
        """Set the nth frame value."""
        self.__nth_frame = value
        print(f"Nth frame: {value}")

    def set_bounding_box_max(self, value):
        """Set the bounding box max value."""
        self.__bbox_max = value
        print(f"Bounding box max: {value}")
