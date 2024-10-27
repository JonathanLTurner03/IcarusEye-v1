import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# TODO add documentation and comments
class VideoStream:
    def __init__(self, source, type, width=1920, fps=60, height=1080, backend=cv2.CAP_DSHOW):
        self.cap = None
        self.source = source

        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend

        if type == 'camera':
            self._setup_camera()
        elif type == 'recording':
            self._setup_recording()
        elif type == 'capture_card':
            self.setup_capture_card()


    def _setup_camera(self):
        # Open the camera using DirectShow (Windows)
        cap = cv2.VideoCapture(self.source, self.backend)

        # Check if the camera opened successfully
        if not cap.isOpened():
            print("Failed to open camera.")
        else:
            print("Camera opened successfully with DirectShow.")

        # Set the resolution (1080p)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Optionally limit the frame rate to avoid high processing loads at 1080p
        cap.set(cv2.CAP_PROP_FPS, self.fps)  # Adjust as needed

        # Set the camera to use the MJPEG codec (FOURCC for MJPG)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if not cap.set(cv2.CAP_PROP_FOURCC, fourcc):
            print("Failed to set MJPEG codec.")
            self.cap = None
        else:
            print("MJPEG codec set successfully.")
            self.cap = cap


    def _setup_recording(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logging.error(f"Error: Unable to open video source {self.source}")
        else:
            # Set the resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            logging.error("Warning: FPS is zero, defaulting to 30")
            fps = 30  # Default to 30 FPS if unable to retrieve
        return fps

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return ret, frame
        return None

    def get_frame_position(self):
        return self.cap.get_frame_postition()

    def release(self):
        self.cap.release()