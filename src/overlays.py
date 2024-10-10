import cv2
import logging
import yaml

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
log_level = config['logging']['level']

logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def draw_boxes(frame, boxes, track_ids, box_color=(0, 255, 0), text_color=(255, 255, 255), margin=5):
    """
    Draw bounding boxes and tracking IDs on the frame.
    :param frame: The video frame (numpy array)
    :param boxes: List of bounding boxes (x1, y1, x2, y2)
    :param track_ids: List of tracking IDs
    :param box_color: Color of the bounding box (default: green)
    :param text_color: Color of the text (default: white)
    :param margin: Margin to apply to the bounding boxes (default: 5 pixels)
    :return: Frame with the bounding boxes and tracking IDs drawn
    """
    frame_height, frame_width = frame.shape[:2]

    for box, track_id in zip(boxes, track_ids):
        # Ensure the bounding box coordinates are integers
        x1, y1, x2, y2 = map(int, box)

        # Apply margin to the bounding box coordinates
        x1 = max(0, x1 + margin)
        y1 = max(0, y1 + margin)
        x2 = min(frame_width, x2 - margin)
        y2 = min(frame_height, y2 - margin)

        # Check if the bounding box is within the frame dimensions
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            logging.warning(f"Bounding box {box} is out of frame bounds")
            continue

        # Debugging: Print box coordinates
        logging.debug(f"Drawing tracked box: ({x1}, {y1}, {x2}, {y2}) with track ID {track_id}")

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Prepare label text with track ID
        label = f"ID: {track_id}"

        # Draw the label above the bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return frame