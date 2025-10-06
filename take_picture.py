from picamera2 import Picamera2
import time

# Initialize the camera
picam2 = Picamera2()

# Start the camera
picam2.start()
time.sleep(2)  # Allow camera to adjust

# Capture an image
image = picam2.capture_array()
from PIL import Image
im = Image.fromarray(image)
im.save("/home/jetson/photo.jpg")

print("Photo captured!")

# Stop the camera
picam2.stop()
