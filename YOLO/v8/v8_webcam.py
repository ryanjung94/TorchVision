from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolov8m-seg.pt")

CONFIDENCE_THRESHOLD = 0.6

# Predict with the model
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Cam Error')
        break
    detection = model(frame)
    results = detection[0].plot()
    cv2.imshow('frame', results)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()