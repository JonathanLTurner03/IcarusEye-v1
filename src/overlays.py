import cv2
import logging
import yaml
import numpy as np

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
log_level = config['logging']['level']

logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


def draw_boxes(frame, boxes, scores, classes, class_names, colors, confidence_threshold=0.5, text_color=(255, 255, 255), thickness=3, alpha=0.3, max_labels=5):
    """
    Draw bounding boxes with class-specific colors and solid corners.
    :param frame: The video frame (numpy array)
    :param boxes: List of bounding boxes (x1, y1, x2, y2)
    :param scores: List of confidence scores
    :param classes: List of class IDs
    :param class_names: Dictionary of class names
    :param colors: Dictionary mapping class IDs to colors (BGR format)
    :param confidence_threshold: Minimum confidence score to display the box
    :param text_color: Color of the text (default: white)
    :param thickness: Thickness of the corner lines
    :param alpha: Transparency level for the overlay (default is 0.3)
    :param max_labels: Maximum number of labels to display
    :return: Frame with bounding boxes and labels drawn
    """
    # Filter and sort detections by confidence score (highest to lowest)
    detections = [(box, score, class_id) for box, score, class_id in zip(boxes, scores, classes) if score >= confidence_threshold]
    detections = sorted(detections, key=lambda x: x[1], reverse=True)  # Sort by confidence

    # Limit the number of labels displayed
    detections = detections[:max_labels]

    for (box, score, class_id) in detections:
        x1, y1, x2, y2 = map(int, box)

        # Debugging: Print box coordinates
        logging.debug(f"Drawing box: ({x1}, {y1}, {x2}, {y2}) with score {score}")

        # Choose box color based on class ID, or default to green if no color is provided
        box_color = colors.get(int(class_id), [0, 255, 0])  # Default to green if no color found

        # Create overlay
        overlay = frame.copy()

        # Create a solid color rectangle with the same alpha value applied to all areas
        overlay[y1:y2, x1:x2] = (np.array(box_color) * alpha + frame[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)

        # Combine overlay with the original frame
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw solid corners instead of full rectangle
        corner_len = min((x2 - x1), (y2 - y1)) // 5  # Corner length is 1/5 of the box's smallest dimension

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, thickness)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, thickness)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, thickness)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, thickness)

        # Prepare label text
        class_name = class_names.get(int(class_id), f"Class {int(class_id)}")
        label = f"{int(class_id)}: {score:.2f}"

        # Debugging: Print the label
        logging.debug(f"Label: {label}")

        # Draw the label above the bounding box
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_w, label_h = label_size

        # Background for text (semi-transparent)
        cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), box_color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return frame



