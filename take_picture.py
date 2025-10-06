import cv2

# GStreamer pipeline for Jetson Nano CSI camera
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, framerate=(fraction)30/1, format=(string)NV12 ! "
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
    # Optional: rotate if upside down
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imwrite("/home/pxl/photo.jpg", frame)
    print("Photo captured!")
else:
    print("Failed to capture frame")

cap.release()
cv2.destroyAllWindows()
