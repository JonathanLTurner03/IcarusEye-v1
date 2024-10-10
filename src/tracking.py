import yaml
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

logging.basicConfig(level=logging.DEBUG)

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class Tracker:
    def __init__(self):
        # Load model path and tracker parameters
        embedder_model_name = config['model']['deepsort']
        tracker_cfg = config['tracker']

        # Initialize DeepSORT tracker with appropriate parameters
        self.tracker = DeepSort(
            max_age=tracker_cfg['max_age'],
            n_init=tracker_cfg['n_init'],
            max_cosine_distance=tracker_cfg['max_cosine_distance'],
            nn_budget=tracker_cfg['nn_budget'],
            embedder_model_name=embedder_model_name,  # Path to the DeepSORT embedder model
            override_track_class=None,       # Optional: Override default tracking classes
            embedder_gpu=True,               # Use GPU for embedding (set to False if not available)
            half=True,                       # Use FP16 for embeddings (set to False if not supported)
            bgr=True,                        # Input frames are in BGR format
        )

    def update_tracks(self, boxes, scores, classes, frame):
        """
        Update tracker with detections from YOLOv8.
        :param boxes: Bounding boxes (x1, y1, x2, y2) as a list of lists
        :param scores: Confidence scores as a list
        :param classes: Class labels as a list
        :param frame: The current frame (numpy array)
        :return: List of tracked objects with unique IDs
        """
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Empty or invalid frame passed to DeepSORT")
            return []

        frame_height, frame_width = frame.shape[:2]

        # Prepare detections for DeepSort
        detections = []
        for box, score, class_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box

            # Ensure the bounding box is valid and within frame boundaries
            if (x2 - x1) > 0 and (y2 - y1) > 0 and x1 >= 0 and y1 >= 0 and x2 <= frame_width and y2 <= frame_height:
                detections.append((box, score, str(class_id)))
            else:
                print(f"Skipping invalid box: ({x1}, {y1}, {x2}, {y2}) outside frame bounds or zero area")

        # If no valid detections, skip tracking
        if len(detections) == 0:
            print("No valid detections to track")
            return []

        print(f"Sending {len(detections)} valid detections to DeepSORT")

        # Update the tracker with valid detections
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Extract tracked objects with bounding boxes and IDs
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            tracked_objects.append({
                'track_id': track_id,
                'bbox': ltrb,
                'class_id': track.get_det_class()
            })

        print(f"DeepSORT returned {len(tracked_objects)} tracked objects")

        return tracked_objects
