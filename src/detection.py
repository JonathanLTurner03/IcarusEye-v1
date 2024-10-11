from ultralytics import YOLO
import torch
import torchvision.transforms as transforms


# TODO: Add documentation and comments
class YOLOv8Detection:
    def __init__(self, model_path, verbose=False):
        """
        Initialize the YOLOv8 model with the given path.
        :param model_path: Path to the YOLOv8 weights file (e.g., custom-pretrained-yolov8s.pt)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.model = YOLO(model_path, verbose=verbose).to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])


    def detect(self, frame, verbose=False):
        frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model(frame_tensor, verbose=verbose)
        boxes, scores, classes = self.process_detections(results)
        return boxes, scores, classes


    def process_detections(self, results):
        boxes = []
        scores = []
        classes = []
        for result in results:
            if result.boxes is not None:
                boxes.extend(result.boxes.xyxy.cpu().numpy())
                scores.extend(result.boxes.conf.cpu().numpy())
                classes.extend(result.boxes.cls.cpu().numpy())
        return boxes, scores, classes
