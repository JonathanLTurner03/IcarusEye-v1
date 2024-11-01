from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QScrollArea
from src.ui.config_panel import ConfigPanel
from src.ui.video_panel import VideoPanel
from src.video_stream import VideoStream
import yaml


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the configuration file
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Setup properties
        self.fps = 0
        self.native_fps = 0
        self.confidence = 50
        self.__device_id = None
        self.__nth_frame = 1
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
        self.scroll_area = QScrollArea()

        # Create and add the ConfigPanel to the layout
        self.config_panel = ConfigPanel(self)
        self.video_panel = VideoPanel(self)

        # Set default values in config.
        self.config_panel.set_fps(1)
        self.config_panel.set_confidence(50)

        self.scroll_area.setWidget(self.config_panel)
        self.scroll_area.setWidgetResizable(True)

        self.layout.addWidget(self.video_panel)
        self.layout.addWidget(self.scroll_area)

        # Set the stretch factors for the layout
        self.layout.setStretch(0, 7)  # VideoPanel takes 70% of the space
        self.layout.setStretch(1, 3)  # ConfigPanel takes 30% of the space

    def set_confidence(self, value):
        """Set the confidence threshold value."""
        self.video_panel.update_confidence_threshold(value / 100)

    # Sets the resolution multiplier
    def set_video_file(self, file_path):
        """Set the video file path."""
        cap = VideoStream(file_path, 'recording')
        self.fps = cap.get_fps()
        self.native_fps = self.fps
        self.video_panel.set_video_stream(cap)

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
        self.video_panel.update_nth_frame(value)
        self.__nth_frame = value
        print(f"Nth frame: {value}")

    def set_bounding_box_max(self, value):
        """Set the bounding box max value."""
        self.__bbox_max = value
        self.video_panel.update_max_boxes(value)

    def set_video_device(self, device):
        """Set the video device."""
        self.__device_id = device
        if device != -1:
            self.video_panel.set_video_stream(VideoStream(device, 'camera'))

        if device == -1:
            # Clear the video device settings.
            print("Video device removed.")
