from picamera import PiCamera
from time import sleep

# Initialize the camera
camera = PiCamera()

# Optional: set camera resolution
camera.resolution = (1920, 1080)

# Start the camera preview
camera.start_preview()
sleep(2)  # Allow the camera to adjust to lighting

# Capture an image
camera.capture('/home/jetson/photo.jpg')
print("Photo captured!")

# Stop the preview
camera.stop_preview()

# Close the camera
camera.close()
