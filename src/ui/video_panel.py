import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSlider, QHBoxLayout, QLabel,
                             QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO
from src.video_stream import VideoStream
from src.opengl_video_widget import OpenGLVideoWidget
import OpenGL.GL as gl
import cupy as cp
import torch
import yaml


def format_time(seconds):
    """Convert seconds to minutes:seconds format."""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"


class VideoPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        # Load the configuration file
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Layout for the video panel
        self.layout = QVBoxLayout(self)
        self.__timer = None

        # Setup attribute variables and layouts.
        self.__opengl_widget = OpenGLVideoWidget(self)
        self.__opengl_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.__opengl_container = QWidget(self)
        self.__opengl_container.setLayout(QVBoxLayout())
        self.__opengl_container.layout().addWidget(self.__opengl_widget)

        self.__control_layout = QVBoxLayout()

        self.__control_buttons_layout = QHBoxLayout()
        self.__control_layout.addLayout(self.__control_buttons_layout)

        self.__play_pause_button = QPushButton("Play", self)
        self.__play_pause_button.clicked.connect(self.__toggle_play_pause)
        self.__control_buttons_layout.addWidget(self.__play_pause_button)

        self.__stop_button = QPushButton("Stop", self)
        self.__stop_button.clicked.connect(self.__stop_video)
        self.__control_buttons_layout.addWidget(self.__stop_button)

        self.__timeline_layout = QVBoxLayout()

        self.__timeline_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.__timeline_slider.setRange(0, 100)
        self.__timeline_slider.sliderMoved.connect(self.__seek_video)
        self.__timeline_layout.addWidget(self.__timeline_slider)

        self.__control_layout.addLayout(self.__timeline_layout)

        self.__current_duration_label = QLabel(format_time(0))
        self.__current_duration_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.__video_duration_label = QLabel(format_time(0))
        self.__video_duration_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        time_layout = QHBoxLayout()
        time_layout.addWidget(self.__current_duration_label, alignment=Qt.AlignmentFlag.AlignLeft)
        time_layout.addWidget(self.__video_duration_label, alignment=Qt.AlignmentFlag.AlignRight)
        self.__timeline_layout.addLayout(time_layout)

        # Add widgets to the main layout with stretch factors
        self.layout.addWidget(self.__opengl_container, stretch=7)
        self.layout.addLayout(self.__control_layout, stretch=3)

        self.__video_stream = None
        self.__is_playing = False
        self.__fps = 30  # Default FPS

        model_path = self.config['model']['yolov8s']
        verbose = self.config['logging']['detection_verbose']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.model = YOLO(model_path, verbose=verbose).to(self.device)

        self.__confidence_threshold = 0
        self.__max_boxes = 5
        self.__frame_counter = 0
        self.__nth_frame = 1
        self.__bounding_boxes = []
        self.__confidences = []
        self.__classes = []
        self.__min_box_size = 15
        self.__class_colormap = \
            {
                0: [0, 1, 1],
                1: [0, 1, 1],
                2: [0, 1, 0],
                3: [1, 0, 0],
                4: [1, 0, 0],
                5: [1, 0, 0],
                6: [0, 1, 0],
                7: [0, 1, 0],
                8: [1, 1, 0],
                9: [1, 1, 1]
            }
        print(self.model.names)

    def set_video_stream(self, video_stream: VideoStream):
        """Set the video stream."""
        self.__video_stream = video_stream
        self.__fps = self.__video_stream.get_fps()
        self.__timeline_slider.setRange(0, 1)
        self.__timeline_slider.setVisible(True)
        self.__timer = QTimer(self)
        self.__timer.timeout.connect(self.update_frame)
        self.__timer.start(int(1000/self.__fps))
        self.__play_video()
        #self.__video_duration_label.setText(format_time(self.__video_stream.frame_count / self.__fps))

    def get_video_stream(self):
        """Get the video stream."""
        return self.__video_stream

    def __toggle_play_pause(self):
        """Toggle between play and pause."""
        if self.__is_playing:
            self.__pause_video()
        else:
            self.__play_video()

    def __play_video(self):
        """Play the video."""
        if self.__video_stream is not None:
            self.__is_playing = True
            self.__play_pause_button.setText("Pause")
            self.__timer.start(1000 // int(self.__fps))
        else:
            QMessageBox.warning(self, "Error", "No video or device selected.")

    def __pause_video(self):
        """Pause the video."""
        if self.__video_stream is not None:
            self.__is_playing = False
            self.__play_pause_button.setText("Play")
            self.__timer.stop()

    def __stop_video(self):
        """Stop the video."""
        self.__is_playing = False
        self.__play_pause_button.setText("Play")
        self.__timer.stop()
        if self.__video_stream is not None:
            self.__video_stream.release()
        self.__video_stream = None

    def __seek_video(self, position):
        """Seek to a specific position in the video."""
        if self.__video_stream is not None:
            frame_number = int(position * self.__video_stream.frame_count / 100)
            self.__video_stream.set_frame_position(frame_number)
            self.update_frame()

    def update_frame(self):
        result = self.__video_stream.get_frame()
        if result:
            ret, frame = result
            self.__frame_counter += 1
        else:
            return

        if self.__frame_counter % self.__nth_frame != 0:
            self.__opengl_widget.upload_frame_to_opengl(cp.asnumpy(frame), self.__bounding_boxes, self.__confidences,
                                                        self.__classes)
            return

        if ret:
            frame_gpu = cp.asnumpy(frame)
            results = self.model(frame_gpu, verbose=False)

            bblist = []
            conflist = []
            clslist = []

            bounding_boxes = []
            confidences = []
            classes = []

            for result in results:
                for box in result.boxes:
                    conf = box.conf.item()
                    if conf >= self.__confidence_threshold:
                        bbox = cp.asarray(box.xyxy[0])
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]

                        # Filter based on bounding box size
                        if width >= self.__min_box_size and height >= self.__min_box_size:
                            bblist.append(bbox)
                            conflist.append(conf)
                            clslist.append(self.__class_colormap[int(box.cls.item())])

            if len(bblist) > 0:
                boxes_array = cp.stack(bblist)
                conf_array = cp.array(conflist)
                cls_array = cp.array(clslist)

                sorted_indicies = cp.argsort(-conf_array)
                boxes_array = boxes_array[sorted_indicies]
                conf_array = conf_array[sorted_indicies]
                cls_array = cls_array[sorted_indicies]

                boxes_array = boxes_array[:self.__max_boxes]
                conf_array = conf_array[:self.__max_boxes]
                cls_array = cls_array[:self.__max_boxes]

                bounding_boxes = cp.asnumpy(boxes_array).tolist()
                confidences = cp.asnumpy(conf_array).tolist()
                classes = cp.asnumpy(cls_array).tolist()


            self.__bounding_boxes = bounding_boxes
            self.__confidences = confidences
            self.__classes = classes
            self.__opengl_widget.upload_frame_to_opengl(frame_gpu, bounding_boxes, confidences, classes)

    def load_video_file(self, file_path):
        """Load a video file."""
        self.__video_stream = VideoStream(file_path, 'recording')
        self.__fps = self.__video_stream.get_fps()
        self.__timeline_slider.setRange(0, 1)
        self.__timeline_slider.setVisible(True)
        self.__video_duration_label.setText(format_time(1))
        self.__play_video()

    def update_confidence_threshold(self, value):
        """Update the confidence threshold."""
        self.__confidence_threshold = value

    def update_max_boxes(self, value):
        """Update the maximum number of boxes."""
        self.__max_boxes = value

    def update_nth_frame(self, value):
        """Update the nth frame."""
        self.__nth_frame = value
