import cv2

# GStreamer pipeline for Jetson Nano camera
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)

# Open the camera
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Capture a frame
ret, frame = cap.read()
if ret:
    cv2.imwrite('/home/pxl/photo.jpg', frame)
    print("Photo captured!")
else:
    print("Failed to capture image")

# Release the camera
cap.release()
cv2.destroyAllWindows()
