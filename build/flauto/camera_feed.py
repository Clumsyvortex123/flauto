import cv2
import argparse

class CameraFeed:
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        """Initialize the camera feed with given settings.

        Args:
            camera_index (int): The index of the camera (usually 0 for default webcam).
            width (int): Width of the camera feed.
            height (int): Height of the camera feed.
            fps (int): Frames per second for the camera feed.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        # Initialize the camera
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise Exception(f"Could not open video device at index {self.camera_index}")

        # Set the resolution and frame rate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def update_settings(self):
        """Allow the user to update camera settings interactively."""
        print("Press 'q' to quit.")
        print("Press 'r' to reset to default resolution and FPS.")
        print("Press 'u' to update camera settings.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Display the current frame
            cv2.imshow('Camera Feed', frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Quit the program
                break
            elif key == ord('r'):  # Reset settings to default
                self.reset_settings()
            elif key == ord('u'):  # Update settings
                self.prompt_for_settings()

        self.cap.release()
        cv2.destroyAllWindows()

    def reset_settings(self):
        """Reset the camera to default settings."""
        print("Resetting to default resolution and FPS...")
        self.width = 640
        self.height = 480
        self.fps = 30
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def prompt_for_settings(self):
        """Prompt the user for new camera settings."""
        print("Enter new camera settings:")
        try:
            new_width = int(input("Width (default 640): ") or self.width)
            new_height = int(input("Height (default 480): ") or self.height)
            new_fps = int(input("FPS (default 30): ") or self.fps)
        except ValueError:
            print("Invalid input, settings not updated.")
            return

        self.width = new_width
        self.height = new_height
        self.fps = new_fps

        # Apply the new settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        print(f"Updated settings: Resolution={self.width}x{self.height}, FPS={self.fps}")

if __name__ == '__main__':
    # Use argparse to get user inputs
    parser = argparse.ArgumentParser(description="Run a live camera feed with adjustable settings.")
    parser.add_argument('--camera_index', type=int, default=0, help="Index of the camera (default: 0)")
    parser.add_argument('--width', type=int, default=640, help="Width of the camera feed (default: 640)")
    parser.add_argument('--height', type=int, default=480, help="Height of the camera feed (default: 480)")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second (default: 30)")

    args = parser.parse_args()

    # Create an instance of CameraFeed with parsed arguments
    try:
        camera = CameraFeed(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        camera.update_settings()
    except Exception as e:
        print(f"Error: {e}")
