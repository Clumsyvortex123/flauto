import cv2
import os
import numpy as np
import argparse
import yaml
from ultralytics import YOLO

class Trainer:
    def __init__(self):
        self.available_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

    def load_model(self, model_version):
        if model_version not in self.available_models:
            raise ValueError(f"Invalid model version. Choose from {self.available_models}")
        return YOLO(model_version)

    def preprocess_images(self, input_folder):
        """
        Preprocess all images in the given folder.
        Steps include grayscale conversion, histogram equalization, and normalization.
        """
        if not os.path.exists(input_folder):
            raise ValueError(f"Folder '{input_folder}' does not exist.")

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for image_file in image_files:
            input_path = os.path.join(input_folder, image_file)
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Warning: Could not read image {input_path}. Skipping...")
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            normalized_image = cv2.normalize(equalized_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            cv2.imwrite(input_path, normalized_image)
            print(f"Processed and saved: {input_path}")

    def update_yaml_paths(self, yaml_file_path, new_train_path, new_test_path, new_val_path):
        """
        Update the dataset paths in a YAML file while preserving order.
        """
        try:
            if not os.path.exists(yaml_file_path):
                print(f"Error: YAML file not found at {yaml_file_path}")
                return

            with open(yaml_file_path, 'r') as file:
                data = yaml.safe_load(file) or {}

            # Ensure train, val, and test paths are updated and placed at the top
            ordered_data = {
                "train": new_train_path,
                "val": new_val_path,
                "test": new_test_path
            }

            # Retain other keys (like nc, names) from the original file
            for key, value in data.items():
                if key not in ordered_data:
                    ordered_data[key] = value  # Preserve other dataset settings

            # Write back to YAML, ensuring correct order
            with open(yaml_file_path, 'w') as file:
                yaml.dump(ordered_data, file, default_flow_style=False, sort_keys=False)

            print(f"✅ Successfully updated dataset paths in {yaml_file_path}")

        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")

    def train_model(self, model_version, yaml_path, project, name):
        """
        Train the object detection model using the given YAML dataset configuration.
        """
        if model_version not in self.available_models:
            raise ValueError(f"Invalid model version. Choose from {self.available_models}")

        model = YOLO(model_version)  # Load model dynamically within this function

        model.train(
            data=yaml_path,
            epochs=5,
            batch=32,
            imgsz=640,
            patience=5,
            project=project,
            name=name,
            pretrained=True,
            optimizer="auto",

            # Data augmentation parameters
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,

            # Model evaluation parameters
            val=True,
            iou=0.7,
            conf=0.25,
            max_det=300,

            # Training strategies
            warmup_epochs=0.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            close_mosaic=10,

            # Miscellaneous
            seed=0,
            verbose=True
        )

        results = model.val()
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Parser for preprocessing images
    parser_preprocess = subparsers.add_parser("preprocess", help="Preprocess images in a folder")
    parser_preprocess.add_argument("--input-folder", type=str, required=True, help="Path to input image folder")

    # Parser for updating YAML dataset paths
    parser_update_yaml = subparsers.add_parser("update_yaml", help="Update dataset paths in a YAML file")
    parser_update_yaml.add_argument("--yaml-path", type=str, required=True, help="Path to dataset YAML file")
    parser_update_yaml.add_argument("--train-path", type=str, required=True, help="New training dataset path")
    parser_update_yaml.add_argument("--val-path", type=str, required=True, help="New validation dataset path")
    parser_update_yaml.add_argument("--test-path", type=str, required=True, help="New test dataset path")

    # Parser for training the model
    parser_train = subparsers.add_parser("train", help="Train the YOLOv8 model")
    parser_train.add_argument("--model-version", type=str, required=True, help="YOLO model version (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser_train.add_argument("--yaml-path", type=str, required=True, help="Path to dataset YAML file")
    parser_train.add_argument("--project", type=str, required=True, help="Project directory to save results")
    parser_train.add_argument("--name", type=str, required=True, help="Experiment name")

    args = parser.parse_args()
    trainer = Trainer()

    if args.command == "preprocess":
        trainer.preprocess_images(args.input_folder)

    elif args.command == "update_yaml":
        trainer.update_yaml_paths(args.yaml_path, args.train_path, args.test_path, args.val_path)

    elif args.command == "train":
        trainer.train_model(args.model_version, args.yaml_path, args.project, args.name)
    else:
        parser.print_help()

