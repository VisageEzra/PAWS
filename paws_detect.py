import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 


cap = cv2.VideoCapture(0)

# List of COCO class IDs for animals
# 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 
# 20: elephant, 21: bear, 22: zebra, 23: giraffe, 14: bird
ANIMAL_CLASSES = 15, 16, 17, 18, 19, 20, 21, 22, 23, 14

print("PAWS Animal Detection System Active...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 
    results = model(frame, stream=True)
    
    animal_found = False

    for r in results:
        
        for box in r.boxes:
            class_id = int(box.cls)
            
            if class_id in ANIMAL_CLASSES:
                animal_found = True
                
        annotated_frame = r.plot()

    # NI YANG BAGI CRASH
    if animal_found:
        
        width = frame.shape
        
        
        cv2.rectangle(annotated_frame, (0, 0), (width, 80), (0, 0, 255), -1)
        
        #  Warning Text 
        cv2.putText(annotated_frame, "WARNING: BINATANG!", (50, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("PAWS - Predictive Animal Warning System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()