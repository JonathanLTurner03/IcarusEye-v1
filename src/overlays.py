import cv2
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def draw_boxes(frame, boxes, scores, classes, class_names, confidence_threshold=0.5, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    """
    Draw bounding boxes and labels on the frame.
    :param frame: The video frame (numpy array)
    :param boxes: List of bounding boxes (x1, y1, x2, y2)
    :param scores: List of confidence scores
    :param classes: List of class IDs
    :param class_names: Dictionary of class names
    :param confidence_threshold: Minimum confidence score to display the box
    :param box_color: Color of the bounding box (default: green)
    :param text_color: Color of the text (default: white)
    :return: Frame with the bounding boxes and labels drawn
    """
    for (box, score, class_id) in zip(boxes, scores, classes):
        if score >= confidence_threshold:
            x1, y1, x2, y2 = box
            # Debugging: Print box coordinates
            logging.log(f"Drawing box: ({x1}, {y1}, {x2}, {y2}) with score {score}")

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Prepare label text
            class_name = class_names.get(int(class_id), f"Class {int(class_id)}")
            label = f"{class_name}: {score:.2f}"

            # Debugging: Print the label
            logging.log(f"Label: {label}")

            # Draw the label above the bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return frame
