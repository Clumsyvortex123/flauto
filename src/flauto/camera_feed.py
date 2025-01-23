#!/usr/bin/env python3

import cv2
import argparse
import logging
import sys


class CameraFeed:
    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CameraFeed")

    def start(self):
        self.logger.info(f"Initializing camera {self.camera_index}...")
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.logger.error(f"Failed to access camera {self.camera_index}")
            raise ValueError("Camera could not be opened.")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        self.logger.info("Starting camera feed. Press 'q' to quit.")

        while True:
            ret, frame = self.capture.read()
            if not ret:
                self.logger.warning("Failed to read frame from the camera.")
                break

            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("Exiting camera feed.")
                break

        self.release()

    def release(self):
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()
        self.logger.info("Camera resources released.")


def main():
    parser = argparse.ArgumentParser(
        description="Camera Feed Application",
        epilog="Example: flauto-camerafeed --camera_index 0 --width 1280 --height 720 --fps 30"
    )
    parser.add_argument(
        "--camera_index", 
        type=int, 
        default=0, 
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=640, 
        help="Frame width in pixels (default: 640)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=480, 
        help="Frame height in pixels (default: 480)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30, 
        help="Frames per second (default: 30)"
    )

    # Show help and exit if no arguments or invalid arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as exc:
        print(f"Error: {exc}\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Initialize and start the CameraFeed
    try:
        camera_feed = CameraFeed(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        camera_feed.start()
    except ValueError as e:
        print(f"Error: {e}\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
