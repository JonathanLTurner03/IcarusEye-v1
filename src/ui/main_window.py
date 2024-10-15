import time
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from src.ui.config_panel import ConfigPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Simulate a delay for loading resources
        time.sleep(3)

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

    def set_fps(self, value):
        """Set the FPS value."""
        # TODO: Implement the update to the video player
        self.fps = value

    def set_confidence(self, value):
        """Set the confidence threshold value."""
        # TODO: Implement the update to the detection worker
        self.confidence = value
