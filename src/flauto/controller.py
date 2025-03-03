import cv2
import time
import threading
import csv
from datetime import datetime
import Jetson.GPIO as GPIO

# --- GPIO Setup ---
# Define the GPIO pins for the 4 relays (adjust these as needed for your hardware)
relay_pins = {
    1: 17,  # Relay for x: 0 to 25
    2: 18,  # Relay for x: 25 to 50
    3: 27,  # Relay for x: 50 to 75
    4: 22   # Relay for x: 75 to 100
}

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
for pin in relay_pins.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Dictionary to track active relay timers (so we can cancel/restart if new data comes in)
active_relay_timers = {}

# --- Helper Functions ---

def get_active_relays(bbox):
    """
    Given a bounding box (4 corners: x1,y1, x2,y2, x3,y3, x4,y4) where x coordinates
    are assumed to be normalized [0,100], determine which quarters are intersected.
    Returns a list of relay numbers.
    """
    xs = [bbox[0], bbox[2], bbox[4], bbox[6]]
    x_min = min(xs)
    x_max = max(xs)
    active_relays = []
    # Define the x-axis quarters (assumed in percentage)
    quarters = {
        1: (0, 25),
        2: (25, 50),
        3: (50, 75),
        4: (75, 100)
    }
    for relay, (start, end) in quarters.items():
        # Check for any overlap between bounding box and quarter section
        if x_min < end and x_max > start:
            active_relays.append(relay)
    return active_relays

def activate_relay(relay, duration):
    """
    Activate the relay (set its GPIO high) and start a timer that will turn it off
    after the specified duration (in seconds).
    """
    GPIO.output(relay_pins[relay], GPIO.HIGH)
    print(f"[{datetime.now()}] Relay {relay} ACTIVATED for {duration:.2f} sec")
    
    def deactivate():
        GPIO.output(relay_pins[relay], GPIO.LOW)
        print(f"[{datetime.now()}] Relay {relay} DEACTIVATED")
        active_relay_timers.pop(relay, None)
    
    timer = threading.Timer(duration, deactivate)
    timer.start()
    active_relay_timers[relay] = timer

def process_detection(bbox, speed, image):
    """
    Process a single detection:
      - Save the image and log the bounding box coordinates.
      - Determine which relays (i.e. x-axis quarters) the detection covers.
      - For each relevant relay, activate it for a duration calculated by:
            duration = (object height) / speed.
        (Here object height is taken as the difference between the maximum and minimum y values.)
    """
    # Save the image with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"image_{timestamp}.jpg"
    cv2.imwrite(image_filename, image)
    
    # Append detection data to a CSV file (timestamp, bbox coordinates, image filename)
    with open("detections.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp] + list(bbox) + [image_filename])
    
    # Compute the vertical span (object height)
    ys = [bbox[1], bbox[3], bbox[5], bbox[7]]
    y_min = min(ys)
    y_max = max(ys)
    object_height = y_max - y_min
    
    # Calculate the time duration for which the relay should remain active.
    # (Assuming speed is in the same y-unit per second as the bbox coordinates.)
    duration = object_height / speed if speed != 0 else 0
    print(f"[{datetime.now()}] Object height: {object_height}, Speed: {speed} => Relay duration: {duration:.2f} sec")
    
    # Determine which relays (x-axis quarters) should be active
    relays_to_activate = get_active_relays(bbox)
    print(f"[{datetime.now()}] Active relay sections based on x-range: {relays_to_activate}")
    
    # Activate each relevant relay. If already active, cancel its timer and restart.
    for relay in relays_to_activate:
        if relay in active_relay_timers:
            active_relay_timers[relay].cancel()
        activate_relay(relay, duration)

# --- Main Routine ---

def main():
    """
    Main routine: For demonstration, we load a sample image and simulate a detection.
    In a real-world scenario, this function would be called repeatedly (or in a loop)
    as new frames and detections become available.
    """
    # Load an image (replace 'sample.jpg' with your image source or camera capture)
    image = cv2.imread("sample.jpg")
    if image is None:
        print("Error: Could not load the image. Please check the file path.")
        return

    # Example bounding box coordinates (x1,y1, x2,y2, x3,y3, x4,y4).
    # Here, we assume x coordinates are normalized (0-100).
    # This example bbox spans from x=10 to x=40 and y=100 to y=200.
    bbox = (10, 100, 40, 100, 40, 200, 10, 200)
    
    # Define the platform’s speed (y-units per second)
    speed = 50  # Adjust as needed
    
    # Process the detection (this will log the data and activate relays)
    process_detection(bbox, speed, image)
    
    # For demonstration, keep the script running until all timers finish.
    # In an actual deployment, you would run this in a loop or integrate into your main control system.
    time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup GPIO on exit
        GPIO.cleanup()
