import cv2


class VideoStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Error: Unable to open video source {source}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        self.cap.release()