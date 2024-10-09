import cv2

class VideoStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Error: Unable to open video source {source}")

    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Warning: FPS is zero, defaulting to 30")
            fps = 60  # Default to 30 FPS if unable to retrieve
        return fps

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        self.cap.release()