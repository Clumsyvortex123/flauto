import zipfile
import json
import cv2
import numpy as np
import io

def load_detections(zip_filename):
    """
    Opens the zip file, reads the metadata.json file, and returns
    the list of detection records.
    """
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        if "metadata.json" not in zipf.namelist():
            raise FileNotFoundError("metadata.json not found in the zip file.")
        metadata_bytes = zipf.read("metadata.json")
        detections = json.loads(metadata_bytes.decode("utf-8"))
    return detections

def extract_image(zip_filename, image_path):
    """
    Extracts and decodes an image from the zip file given its internal path.
    Returns the image as a numpy array.
    """
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        if image_path not in zipf.namelist():
            raise FileNotFoundError(f"Image file {image_path} not found in the zip file.")
        image_data = zipf.read(image_path)
    # Convert the byte data to a numpy array and decode it with OpenCV
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def main():
    zip_filename = input("Enter path to the detection ZIP file: ").strip()
    try:
        detections = load_detections(zip_filename)
    except Exception as e:
        print("Error loading detections:", e)
        return

    if not detections:
        print("No detection records found in metadata.")
        return

    # List available detection timestamps
    print("\nAvailable detection timestamps:")
    for det in detections:
        print(f" - {det['timestamp']}")

    selected_timestamp = input("\nEnter the timestamp you want to view: ").strip()
    selected_detection = None
    for det in detections:
        if det['timestamp'] == selected_timestamp:
            selected_detection = det
            break

    if not selected_detection:
        print("No detection found for that timestamp.")
        return

    # Display detection information
    print("\nDetection Data:")
    for key, value in selected_detection.items():
        print(f"{key}: {value}")

    # Extract and display the image
    image_path = selected_detection.get("image_filename")
    if not image_path:
        print("No image filename found in the selected detection record.")
        return

    try:
        image = extract_image(zip_filename, image_path)
    except Exception as e:
        print("Error extracting image:", e)
        return

    if image is None:
        print("Failed to decode the image.")
        return

    # Display the image using OpenCV
    cv2.imshow(f"Detection - {selected_timestamp}", image)
    print("\nPress any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
