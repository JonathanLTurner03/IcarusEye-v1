import os
import yaml
import shutil
import cv2


def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def convert_visdrone_to_yolo(annotation_file, image_width, image_height):
    """Converts VisDrone annotations to YOLO format."""
    yolo_annotations = []
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # Extract and convert necessary parts
            class_id = int(parts[5]) - 1  # Class ID (convert to 0-based index for YOLO)
            xmin = float(parts[0])
            ymin = float(parts[1])
            width = float(parts[2])
            height = float(parts[3])

            # Convert to YOLO format (normalized values)
            x_center = (xmin + width / 2) / image_width
            y_center = (ymin + height / 2) / image_height
            norm_width = width / image_width
            norm_height = height / image_height

            # Append the converted annotation
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")
    return yolo_annotations


def process_dataset(images_dir, annotations_dir, output_labels_dir, raw_images_dir):
    """Processes a dataset by converting annotations and copying images."""

    # Ensure the necessary directories exist (create them if missing)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Copy images from raw dataset to the target images directory
    for image_file in os.listdir(raw_images_dir):
        if image_file.endswith('.jpg'):
            src_image_path = os.path.join(raw_images_dir, image_file)
            dst_image_path = os.path.join(images_dir, image_file)
            shutil.copy(src_image_path, dst_image_path)
            print(f"Copied image {src_image_path} to {dst_image_path}")

            # Get corresponding annotation file
            annotation_file = os.path.join(annotations_dir, image_file.replace('.jpg', '.txt'))

            if os.path.exists(annotation_file):
                # Read image to get its dimensions
                image = cv2.imread(dst_image_path)
                if image is None:
                    print(f"Error reading image: {dst_image_path}")
                    continue
                image_height, image_width = image.shape[:2]

                # Convert annotations to YOLO format
                yolo_annotations = convert_visdrone_to_yolo(annotation_file, image_width, image_height)

                # Save YOLO annotations to the output labels directory
                yolo_label_path = os.path.join(output_labels_dir, image_file.replace('.jpg', '.txt'))
                with open(yolo_label_path, 'w') as yolo_file:
                    yolo_file.writelines(yolo_annotations)

                print(f"Converted {annotation_file} to {yolo_label_path}")
            else:
                print(f"Annotation file {annotation_file} not found.")


if __name__ == "__main__":
    # Load configuration from config.yaml
    config = load_config('config.yaml')

    # Process the training dataset
    print("Processing training set...")
    process_dataset(
        config['train'],
        config['train_annotations'],
        config['train_labels'],
        config['visdrone-train']
    )

    # Process the validation dataset
    print("Processing validation set...")
    process_dataset(
        config['val'],
        config['val_annotations'],
        config['val_labels'],
        config['visdrone-val']
    )

    # Process the test dataset
    print("Processing test set...")
    process_dataset(
        config['test'],
        config['test_annotations'],
        config['test_labels'],
        config['visdrone-test']
    )
