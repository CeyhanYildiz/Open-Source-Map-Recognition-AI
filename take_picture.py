import cv2

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0, cv2.CAP_GSTREAMER)

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
