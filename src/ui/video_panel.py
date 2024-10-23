from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSlider, QHBoxLayout, QLabel,
                             QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from src.video_stream import VideoStream
from src.opengl_video_widget import OpenGLVideoWidget
import cv2


def format_time(seconds):
    """Convert seconds to minutes:seconds format."""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"


class VideoPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        # Layout for the video panel
        self.layout = QVBoxLayout(self)

        # Setup attribute variables and layouts.
        self.__opengl_widget = OpenGLVideoWidget(self)
        self.__opengl_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        self.layout.addWidget(self.__opengl_widget, stretch=7)
        self.layout.addLayout(self.__control_layout, stretch=3)

        self.__timer = QTimer(self)
        self.__timer.timeout.connect(self.update_frame)

        self.__video_stream = None
        self.__is_playing = False
        self.__fps = 30  # Default FPS

    def set_video_stream(self, video_stream: VideoStream):
        """Set the video stream."""
        self.__video_stream = video_stream
        self.__fps = self.__video_stream.get_fps()
        self.__timeline_slider.setRange(0, 1)
        self.__timeline_slider.setVisible(True)
        #self.__video_duration_label.setText(format_time(self.__video_stream.frame_count / self.__fps))
        self.__play_video()

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
        """Update the OpenGL widget with the current frame."""
        if self.__video_stream:
            frame = self.__video_stream.get_frame()

            #self.__opengl_widget.update_frame(frame)
            if False: # TODO turn on if a video, off if not.
                current_position = int(self.__video_stream.get_frame_position() * 100 / self.__video_stream.frame_count)
                self.__timeline_slider.setValue(current_position)
                self.__current_duration_label.setText(format_time(self.__video_stream.get_frame_position() / self.__fps))

    def load_video_file(self, file_path):
        """Load a video file."""
        self.__video_stream = VideoStream(file_path, 'recording')
        self.__fps = self.__video_stream.get_fps()
        self.__timeline_slider.setRange(0, 1)
        self.__timeline_slider.setVisible(True)
        self.__video_duration_label.setText(format_time(1))
        self.__play_video()