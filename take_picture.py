import cv2

# Open the camera device
cap = cv2.VideoCapture(0)  # /dev/video0

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Capture a single frame
ret, frame = cap.read()
if ret:
    cv2.imwrite("/home/pxl/photo.jpg", frame)
    print("Photo captured!")
else:
    print("Failed to capture frame")

# Release the camera
cap.release()
cv2.destroyAllWindows()
