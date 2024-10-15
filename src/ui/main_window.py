import time
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from src.ui.config_panel import ConfigPanel
import cv2
import os
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Setup properties
        self.fps = 30
        self.native_fps = 30
        self.confidence = 50

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set up the layout for the central widget
        self.layout = QVBoxLayout(self.central_widget)

        # Create and add the ConfigPanel to the layout
        self.config_panel = ConfigPanel(self)
        self.layout.addWidget(self.config_panel)

    # TODO: Implement the following methods
    def get_available_classes(self):
        # Return a list of available classes
        return ["Class1", "Class2", "Class3"]

    def set_fps(self, value):
        """Set the FPS value."""
        # TODO: Implement the update to the video player
        self.fps = value

    def set_confidence(self, value):
        """Set the confidence threshold value."""
        # TODO: Implement the update to the detection worker
        self.confidence = value

    # Helper functions

    # Gets the list of the available video input devices
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

        return devices
