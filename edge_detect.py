import cv2
import numpy as np

cap = cv2.VideoCapture('faces.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(460,320))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Basic thresholds
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(gray, 100, 200)
    
    # Convert to BGR for stacking
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Show side by side
    combined = np.hstack([frame, binary_bgr, edges_bgr])
    
    cv2.imshow('Original | Binary | Edges', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()