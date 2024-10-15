import time
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from src.ui.config_panel import ConfigPanel
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

        # Get the available classes
        self.__class_details = self.config['class_details']
        self.__multi_color_classes = False
        self.__omit_classes = []

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set up the layout for the central widget
        self.layout = QVBoxLayout(self.central_widget)

        # Create and add the ConfigPanel to the layout
        self.config_panel = ConfigPanel(self)
        self.layout.addWidget(self.config_panel)

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

    # Sets the multivalue color classes
    def set_multi_color_classes(self, value):
        """Set the multi-color classes value."""
        self.__multi_color_classes = value
        print(f"Multi-color classes: {value}")



    def populate_device_dropdown(self) -> list:
        """Populate the dropdown with available video input devices."""
        devices = []
        index = 0

        while True:
            # Redirect stderr to suppress camera indexing errors
            sys.stderr = open(os.devnull, 'w')
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            sys.stderr = sys.__stderr__  # Restore stderr

            if not cap.read()[0]:
                break
            devices.append(f"Device {index}")
            cap.release()
            index += 1

        # Update the dropdown in the main thread
        return devices

    def update_omitted_classes(self, classes):
        """Update the omitted classes."""
        self.__omit_classes = classes
