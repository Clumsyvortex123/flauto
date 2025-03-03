import time
import os
import cv2
import json
import zipfile
import threading
from datetime import datetime
from dataclasses import dataclass, asdict

# =============================================================================
# Global Coordinate Conversion
# =============================================================================
class GlobalCoordinateConverter:
    """
    Converts local (x, y) coordinates into global coordinates.
    Global y is computed by adding an offset that increases over time,
    based on a constant speed from the system's start.
    """
    def __init__(self, speed: float, initial_y: float = 0.0):
        """
        :param speed: The speed at which the global y increases (units per second)
        :param initial_y: The starting global y coordinate (default: 0.0)
        """
        self.speed = speed
        self.initial_y = initial_y
        self.start_time = time.time()
    
    def to_global(self, x: float, y: float) -> (float, float):
        """
        Converts a local (x, y) coordinate to a global coordinate.
        :param x: local x-coordinate (unchanged)
        :param y: local y-coordinate; global y is y + speed * elapsed_time
        :return: (global_x, global_y)
        """
        elapsed = time.time() - self.start_time
        global_y = self.initial_y + y + (self.speed * elapsed)
        return (x, global_y)

# =============================================================================
# Data Structure for Detection Data
# =============================================================================
@dataclass
class DetectionData:
    """
    A structure to store detection information.
    - timestamp: Time of detection (string)
    - global_coordinates: List of (x, y) tuples in global coordinate space
    - local_coordinates: List of (x, y) tuples in local coordinate space
    - image_filename: Filename where the associated image is saved
    - additional_inputs: A dictionary for any extra parameters
    """
    timestamp: str
    global_coordinates: list
    local_coordinates: list
    image_filename: str
    additional_inputs: dict

# =============================================================================
# Utility Functions
# =============================================================================
def get_bounding_box_dimensions(coords: list) -> (float, float):
    """
    Computes the width and height of a bounding box given its corner coordinates.
    :param coords: List of (x, y) tuples (for example, the 4 corners of the box)
    :return: (width, height) of the bounding box
    """
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width, height

def package_detection_data(image, local_coords: list, additional_inputs: dict,
                           converter: GlobalCoordinateConverter, image_save_dir="images") -> DetectionData:
    """
    Packages the detection data into a DetectionData instance.
    It converts the local coordinates to global coordinates, saves the image to disk,
    and returns a data structure that encapsulates all the detection information.
    
    :param image: Image as a numpy array (e.g., loaded via OpenCV)
    :param local_coords: List of (x, y) tuples representing local coordinates (e.g., bounding box corners)
    :param additional_inputs: Dictionary of additional parameters or inputs
    :param converter: GlobalCoordinateConverter instance used to obtain global coordinates
    :param image_save_dir: Directory to save the image (will be created if it does not exist)
    :return: DetectionData object containing the packaged data
    """
    # Generate a timestamp for naming and record-keeping
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert local coordinates to global using the provided converter
    global_coords = [converter.to_global(x, y) for (x, y) in local_coords]
    
    # Ensure the directory for saving images exists
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
        
    # Save the image with a timestamp in its filename
    image_filename = os.path.join(image_save_dir, f"image_{timestamp}.jpg")
    cv2.imwrite(image_filename, image)
    
    # Create and return the DetectionData instance
    detection = DetectionData(
        timestamp=timestamp,
        global_coordinates=global_coords,
        local_coordinates=local_coords,
        image_filename=image_filename,
        additional_inputs=additional_inputs
    )
    return detection

def write_detections_to_zip(detections: list, zip_filename="detection_data.zip"):
    """
    Takes a list of DetectionData objects and writes them to a ZIP archive.
    The ZIP file contains a metadata JSON file with all detection details,
    and all associated image files.
    
    :param detections: List of DetectionData objects
    :param zip_filename: Name of the output ZIP file
    """
    # Convert the list of DetectionData objects into a list of dictionaries
    metadata = [asdict(d) for d in detections]
    
    # Create a temporary metadata JSON file
    metadata_filename = "metadata.json"
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Create the zip archive and add the metadata and images
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(metadata_filename)
        for detection in detections:
            if os.path.exists(detection.image_filename):
                zipf.write(detection.image_filename)
    
    # Clean up the temporary metadata file
    os.remove(metadata_filename)
    print(f"Detections written to {zip_filename}")

# =============================================================================
# Example of Using the Module
# =============================================================================
if __name__ == "__main__":
    # Create a GlobalCoordinateConverter with a chosen speed (units per second)
    converter = GlobalCoordinateConverter(speed=10)  # Adjust speed as needed
    
    # Assume you capture an image from a camera (here we load a sample image)
    image = cv2.imread("sample.jpg")
    if image is None:
        raise FileNotFoundError("Sample image not found. Please provide a valid image path.")
    
    # Example local coordinates for a bounding box (list of (x, y) tuples)
    # These might represent the four corners of a detection box.
    local_coords = [(10, 100), (40, 100), (40, 200), (10, 200)]
    
    # Optionally include any additional inputs (e.g., detection confidence, object class)
    additional_inputs = {"confidence": 0.92, "object_class": "vehicle"}
    
    # Package the detection data
    detection = package_detection_data(image, local_coords, additional_inputs, converter)
    
    # For demonstration, we collect one or more detection records in a list.
    detections = [detection]
    
    # Write the collected detections to a zip file.
    write_detections_to_zip(detections)
