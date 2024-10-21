from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider, QPushButton, QCheckBox, QRadioButton,
                             QComboBox, QFileDialog, QSpacerItem, QSizePolicy, QHBoxLayout, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIntValidator
from src.threads import DeviceScanner
import logging
import cv2


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
        self.__performance_group = QGroupBox("Performance Settings")

        # Defines global variables for the configuration panel
        self.__fps_label = None
        self.__confidence_label = None
        self.__omitted_classes = []
        self.__fps_slider = None
        self.__nth_frame = 0
        self.__device_thread = None
        self.__device_worker = None
        self.__refreshing = False
        self.__codec_dropdown = None
        self.__pixel_format_dropdown = None
        self.__codec_label = None
        self.__pixel_format_label = None


        # Initialize the video and detection settings
        self.__init_input()
        self.__init_video()
        self.__init_detection()
        self.__init_performance()

        # Add the config group to the main layout
        self.__config_layout.addWidget(self.__input_settings)
        self.__config_layout.addWidget(self.__video_group)
        self.__config_layout.addWidget(self.__detection_group)
        self.__config_layout.addWidget(self.__performance_group)

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
        self.__device_dropdown.currentIndexChanged.connect(self.__update_selected_device)

        self.__refresh_button = QPushButton("Refresh Devices")
        self.__refresh_button.clicked.connect(self.__refresh_devices)

        # File input button
        self.__file_button = QPushButton("Select Video File")
        self.__file_button.clicked.connect(self.__select_video_file)
        self.__file_button.setEnabled(False)  # Initially disabled

        # Add widgets to the video input layout
        video_input_layout.addWidget(device_radio)
        video_input_layout.addWidget(self.__device_dropdown)
        video_input_layout.addWidget(self.__refresh_button)
        video_input_layout.addWidget(file_radio)
        video_input_layout.addWidget(self.__file_button)

        # Set the layout for the input settings group
        self.__input_settings.setLayout(video_input_layout)
        self.__refresh_devices()

    def __init_video(self):
        # Frame Rate Slider
        self.__fps_label = QLabel(f"Frame Rate (FPS): {self.controller.fps}, Native: {self.controller.native_fps}")
        self.__fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.__fps_slider.setRange(1, 60)
        self.__fps_slider.setValue(self.controller.fps)
        self.__fps_slider.valueChanged.connect(self.__set_fps)

        # Common FPS Buttons
        fps_buttons_layout = QHBoxLayout()
        for fps in [24, 30, 60]:
            button = QPushButton(f"{fps} FPS")
            button.clicked.connect(lambda _, f=fps: self.__set_fps_button(f, self.__fps_slider))
            fps_buttons_layout.addWidget(button)

        # Codec and Pixel Format combined into a single dropdown
        self.__codec_label = QLabel("Codec and Format:")
        self.__codec_dropdown = QComboBox()
        self.__codec_dropdown.currentIndexChanged.connect(self.__update_codec_selection)

        # Resolution Settings
        self.__resolution_label = QLabel("Resolution:")
        self.__resolution_dropdown = QComboBox()
        self.__resolution_dropdown.currentIndexChanged.connect(self.__update_resolution)

        # Add to layout
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.__fps_label)
        video_layout.addWidget(self.__fps_slider)
        video_layout.addLayout(fps_buttons_layout)
        video_layout.addWidget(self.__codec_label)
        video_layout.addWidget(self.__codec_dropdown)
        video_layout.addWidget(self.__resolution_label)
        video_layout.addWidget(self.__resolution_dropdown)

        # Set the layout for the video group
        self.__video_group.setLayout(video_layout)

    def __init_detection(self):
        # Confidence Threshold Slider
        self.__confidence_label = QLabel(f"Confidence Threshold: {self.controller.confidence}")
        self.__confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.__confidence_slider.setRange(0, 100)
        self.__confidence_slider.setValue(self.controller.confidence)

        # Connect the slider's valueChanged signal to the update method
        self.__confidence_slider.valueChanged.connect(self.__update_confidence)

        # Enable/Disable Class-Specific Bounding Boxes
        __class_specific_bbox_checkbox = QCheckBox("Enable Class-Specific Bounding Boxes")
        __class_specific_bbox_checkbox.stateChanged.connect(self.__toggle_class_specific_bbox)

        # Omit Classes Section
        self.__omit_classes_checkbox = QCheckBox("Omit Classes")
        self.__omit_classes_checkbox.stateChanged.connect(self.__toggle_omit_classes)

        self.__classes_dropdown = QComboBox()
        self.__classes_dropdown.addItems(self.controller.get_available_classes())

        self.__add_class_button = QPushButton("Add Class")
        self.__add_class_button.clicked.connect(self.__add_class_to_omit)

        self.__remove_class_button = QPushButton("Remove Class")
        self.__remove_class_button.clicked.connect(self.__remove_class_from_omit)

        self.__omitted_classes_label = QLabel("Omitted Classes: None")

        # Set default states
        self.__remove_class_button.setEnabled(False)
        self.__add_class_button.setEnabled(False)
        self.__classes_dropdown.setEnabled(False)

        # Add to layout
        detection_layout = QVBoxLayout()
        detection_layout.addWidget(self.__confidence_label)
        detection_layout.addWidget(self.__confidence_slider)
        detection_layout.addWidget(__class_specific_bbox_checkbox)
        detection_layout.addWidget(self.__omit_classes_checkbox)
        detection_layout.addWidget(self.__classes_dropdown)
        detection_layout.addWidget(self.__add_class_button)
        detection_layout.addWidget(self.__remove_class_button)
        detection_layout.addWidget(self.__omitted_classes_label)

        # Set the layout for the detection group
        self.__detection_group.setLayout(detection_layout)

    def __init_performance(self):
        # Performance Settings Layout
        performance_layout = QVBoxLayout()

        # nth frame detection
        self.__nth_frame_label = QLabel("Nth Frame Detection:")
        self.__nth_frame_dropdown = QComboBox()
        nth_frames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

        self.__nth_frame_dropdown.addItems(nth_frames)
        self.__nth_frame_dropdown.currentIndexChanged.connect(self.__update_nth_frame)

        # Max Bounding Box
        self.__bounding_box_limit_label = QLabel("Max Bounding Box:")
        self.__bounding_box_limit = QLineEdit()
        self.__bounding_box_limit.setPlaceholderText("Enter max bounding box")
        self.__bounding_box_limit.setValidator(QIntValidator(0, 9999))
        self.__bounding_box_limit.setText("100")  # Set default bounding box size

        # Apply Button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.__apply_performance_settings)

        # Add to layout
        performance_layout.addWidget(self.__nth_frame_label)
        performance_layout.addWidget(self.__nth_frame_dropdown)
        performance_layout.addWidget(self.__bounding_box_limit_label)
        performance_layout.addWidget(self.__bounding_box_limit)
        performance_layout.addWidget(apply_button)

        # Set the layout for the performance group
        self.__performance_group.setLayout(performance_layout)

    def __toggle_class_specific_bbox(self, state):
        """Enable or disable class-specific bounding boxes."""
        enabled = state
        self.controller.set_multi_color_classes(enabled)
        self.__classes_dropdown.clear()
        self.__classes_dropdown.addItems(self.controller.get_available_classes())

    def __toggle_input_type(self):
        """Toggle between device input and file input."""
        if self.__device_dropdown.isEnabled():
            self.__device_dropdown.setEnabled(False)
            self.__refresh_button.setEnabled(False)
            self.__file_button.setEnabled(True)
        else:
            self.__device_dropdown.setEnabled(True)
            if self.__refreshing:
                self.__refresh_button.setEnabled(False)
            else:
                self.__refresh_button.setEnabled(True)
            self.__file_button.setEnabled(False)

    def __select_video_file(self):
        """Open a file dialog to select a video file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.controller.set_video_file(file_name)

    def __refresh_devices(self):
        """Refresh the list of available devices."""
        self.__device_thread = QThread()
        self.__device_worker = DeviceScanner()
        self.__device_worker.moveToThread(self.__device_thread)
        self.__device_thread.started.connect(self.__device_worker.run)
        self.__device_worker.devices_scanned.connect(self.__populate_devices)
        self.__refreshing = True
        self.__refresh_button.setText("Refreshing Devices...")
        self.__refresh_button.setEnabled(False)
        self.__device_thread.start()

    def __populate_devices(self, devices):
        """Populate the dropdown with available video input devices."""
        self.__device_dropdown.clear()
        self.__codec_dropdown.clear()
        self.__resolution_dropdown.clear()
        self.__resolution_label.setText("Resolution: ")
        self.__device_dropdown.addItems(["Select Device"])

        # Populate the dropdown with device names and indices
        for device in devices:
            index = device['index']
            name = device['name']
            self.__device_dropdown.addItem(f"{index}: {name}", device)

        self.__device_thread.quit()
        self.__device_thread.wait()
        self.controller.set_video_device(-1)
        self.__refresh_button.setText("Refresh Devices")
        self.__refreshing = False
        self.__refresh_button.setEnabled(True)

        # Automatically populate codecs if devices are available
        if devices:
            self.__update_selected_device()

    def __update_selected_device(self):
        """Update the selected device based on the dropdown."""
        selected_device_index = self.__device_dropdown.currentIndex()
        if selected_device_index > 0:
            selected_device = self.__device_dropdown.itemData(selected_device_index)
            self.controller.set_video_device(selected_device['index'])
            self.__populate_codecs(selected_device['details'])

    def __populate_codecs(self, details):
        """Populate the codec dropdown based on the selected device's capabilities."""
        self.__codec_dropdown.clear()
        self.__resolution_dropdown.clear()
        self.__resolution_label.setText("Resolution: ")

        if len(details) == 1:
            # If there's only one codec, select it automatically
            self.__codec_dropdown.addItem(details[0]['codec'], details[0])
            self.__codec_dropdown.setCurrentIndex(0)
            self.__populate_resolutions(details[0]['resolutions'])
            self.controller.set_codec(details[0]['codec'])
        else:
            self.__codec_dropdown.addItems(["Select Codec"])
            # If there are multiple codecs, populate the dropdown normally
            for detail in details:
                codec = detail['codec']
                self.__codec_dropdown.addItem(codec, detail)

    def __populate_resolutions(self, resolutions):
        """Populate the resolution dropdown based on the selected codec, sorted by width (highest to lowest)."""
        self.__resolution_dropdown.clear()
        self.__resolution_dropdown.addItems(["Select Resolution"])

        sorted_resolutions = sorted(
            resolutions,
            key=lambda r: int(r['max_resolution'].split('x')[0]),  # Sort by width
            reverse=True
        )

        for res in sorted_resolutions:
            min_res = res['min_resolution']
            max_res = res['max_resolution']
            min_fps = int(res['min_fps'])
            max_fps = int(res['max_fps'])
            resolution_text = f"{max_res} (FPS: {min_fps} - {max_fps})"
            self.__resolution_dropdown.addItem(resolution_text, res)

    def __update_resolution(self, index):
        """Update the resolution based on the selected index."""
        selected_detail = self.__resolution_dropdown.itemData(index)
        if selected_detail:
            min_resolution = selected_detail['min_resolution']
            max_resolution = selected_detail['max_resolution']
            min_fps = int(selected_detail['min_fps'])
            max_fps = int(selected_detail['max_fps'])

            # Set the resolution to the max available resolution and FPS
            self.controller.set_resolution(max_resolution, max_fps)

            # Update the resolution label or any other relevant UI elements
            self.__resolution_label.setText(f"Resolution: {max_resolution}, FPS: {max_fps}")

    def __update_codec_selection(self, index):
        """Update the resolutions based on the selected codec."""
        if index > 0:
            selected_detail = self.__codec_dropdown.itemData(index)
            if selected_detail:
                self.__populate_resolutions(selected_detail['resolutions'])
                self.controller.set_codec(selected_detail['codec'])

    def __toggle_omit_classes(self, state):
        """Enable or disable the omit classes section."""
        enabled = state
        self.__classes_dropdown.setEnabled(enabled)
        self.__add_class_button.setEnabled(enabled)
        self.__remove_class_button.setEnabled(enabled)
        self.__is_live = False
        if not enabled:
            self.__omitted_classes_label.setText("Omitted Classes: (disabled)")
        else:
            self.__omitted_classes_label.setText("Omitted Classes: None")
            self.__update_omitted_classes_label()

    def __add_class_to_omit(self):
        """Add the selected class to the omitted classes list."""
        selected_class = self.__classes_dropdown.currentText().split(':')[0].strip()
        if selected_class and selected_class not in self.__omitted_classes:
            self.__omitted_classes.append(selected_class)
            self.__update_omitted_classes_label()

    def __remove_class_from_omit(self):
        """Remove the selected class from the omitted classes list."""
        selected_class = self.__classes_dropdown.currentText().split(':')[0].strip()
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
        self.controller.update_omitted_classes(self.__omitted_classes)

    def __set_fps_button(self, value, slider):
        """Set the FPS value and update the slider."""
        self.__set_fps(value)
        slider.setValue(value)

    def __set_fps(self, value):
        """Set the FPS value and update the slider."""
        self.__fps_label.setText(f"Frame Rate (FPS): {value}, Native: {self.controller.native_fps}")

    def __update_nth_frame(self, index):
        """Update the nth frame detection based on the selected index."""
        nth_frame = self.__nth_frame_dropdown.itemText(index)
        self.__nth_frame = int(nth_frame)

    def __apply_performance_settings(self):
        """Apply the performance settings."""
        if self.__bounding_box_limit.text() != "":
            max_bounding_box = self.__bounding_box_limit.text()
        else:
            max_bounding_box = 100
            self.__bounding_box_limit.setText("100")

        nth_frame = self.__nth_frame_dropdown.currentText()
        self.controller.set_nth_frame(int(nth_frame))
        self.controller.set_bounding_box_max(int(max_bounding_box))

    def __update_confidence(self, value):
        """Update the label and perform actions when the slider value changes."""
        self.__confidence_label.setText(f"Confidence Threshold: {value}")
        self.controller.set_confidence(value)

    def set_fps(self, value):
        """Update the label and perform actions when the slider value changes."""
        self.__set_fps(value)

    def set_confidence(self, value):
        """Update the label and perform actions when the slider value changes."""
        self.__confidence_slider.setValue(value)
        self.__confidence_label.setText(f"Confidence Threshold: {value}")

    def set_performance_settings(self, nth_frame, max_bounding_box):
        """Update the nth frame and max bounding box settings."""
        self.__nth_frame_dropdown.setCurrentIndex(nth_frame - 1)
        self.__bounding_box_limit.setText(str(max_bounding_box))
