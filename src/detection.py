from ultralytics import YOLO


class YOLOv8Detection:
    def __init__(self, model_path):
        """
        Initialize the YOLOv8 model with the given path.
        :param model_path: Path to the YOLOv8 weights file (e.g., custom-pretrained-yolov8s.pt)
        """
        self.model = YOLO(model_path)

    def detect(self, frame, verbose=False):
        """
        Run YOLOv8 detection on the input frame.
        :param verbose: Use verbose mode for debugging
        :param frame: The current video frame (numpy array)
        :return: A list of detected bounding boxes, class labels, and confidence scores
        """
        # Perform inference on the frame
        results = self.model(frame, verbose=verbose)

        # Extract detections (bounding boxes, class labels, confidence scores)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        classes = results[0].boxes.cls.cpu().numpy()  # Class labels

        return boxes, scores, classes
