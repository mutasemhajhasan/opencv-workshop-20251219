import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255)

# Start video
cap = cv2.VideoCapture("traffic2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Detect objects
    class_ids, scores, boxes = model.detect(frame, confThreshold=0.5)
    # Draw rectangles
    for id,(x, y, w, h) in zip (class_ids,boxes):
        if id==2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 255, 156), 2)
    count=len(boxes)    
    cv2.putText(frame, f"Vehicle Count: {count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Show result
    cv2.imshow("Object Detection", frame)
    
    # Press ESC to quit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()