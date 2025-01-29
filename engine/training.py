import cv2
import os
import numpy as np
import argparse
import yaml
from ultralytics import YOLO

class trainer:
    def __init__(self, model_version='yolov8s'):
        self.available_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        print(f"Available YOLO models: {', '.join(self.available_models)}")
        if model_version not in self.available_models:
            raise ValueError(f"Invalid model version. Choose from {self.available_models}")
        self.model = YOLO(model_version)

    def preprocess_images(self, input_folder):
        """
        Preprocess all images in the given folder.
        Steps include grayscale conversion, histogram equalization, and normalization.
        
        :param input_folder: Path to the folder containing images to preprocess.
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

    def update_yaml_paths(self, yaml_file_path, new_train_path, new_val_path):
        """
        Update the training and validation paths in a dataset YAML file.
        
        :param yaml_file_path: Path to the YAML file.
        :param new_train_path: New path for training dataset.
        :param new_val_path: New path for validation dataset.
        """
        try:
            with open(yaml_file_path, 'r') as file:
                data = yaml.safe_load(file)

            data['train'] = new_train_path if 'train' in data else data.get('train', new_train_path)
            data['val'] = new_val_path if 'val' in data else data.get('val', new_val_path)

            with open(yaml_file_path, 'w') as file:
                yaml.dump(data, file, indent=2)

            print(f"Successfully updated paths in {yaml_file_path}")

        except FileNotFoundError:
            print(f"Error: YAML file not found at {yaml_file_path}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def train_model(self, yaml_path, project, name):
        """
        Train the object detection model using the given YAML dataset configuration.
        
        :param yaml_path: Path to dataset YAML file.
        :param project: Directory to save project outputs.
        :param name: Experiment name.
        """
        self.model.train(
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
        
        results = self.model.val()
        return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the vision system.")
    parser.add_argument('--model-version', type=str, default='yolov8n', help='Enter the version of YOLOv8')
    parser.add_argument('--yaml-path', type=str, required=True, help='Enter the path of the YAML file for the dataset')
    parser.add_argument('--project', type=str, required=True, help='Enter the directory to save the results')
    parser.add_argument('--name', type=str, required=True, help='Enter the name of the folder for results')
    args = parser.parse_args()
    trainer = trainer(model_version=args.model_version)
    trainer.train_model(args.yaml_path, args.project, args.name)
