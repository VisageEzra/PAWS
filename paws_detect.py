import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

last_saved_time = 0
save_cooldown = 5  # Seconds to wait between photos

model = YOLO('yolov8n.pt') 

save_path = "detections"

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

    # 4. THE ACTION: Only trigger if an animal is found
    if animal_found:
        try:
            # Get frame dimensions and force them to integers
            h, w = frame.shape[:2]
            w = int(w)
            
            # --- VISUAL OUTPUT ---
            # Draw the Red Warning Bar (Solid)
            cv2.rectangle(annotated_frame, (0, 0), (w, 100), (0, 0, 255), -1)
            
            # Draw a White Border for "Academic Style"
            cv2.rectangle(annotated_frame, (0, 0), (w, 100), (255, 255, 255), 2)
            
            # Put the Warning Text
            cv2.putText(annotated_frame, "!!! ANIMAL DETECTED !!!", (int(w*0.1), 65), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

            # --- CONSOLE OUTPUT ---
            # This is your "Log" for the supervisor
            print("LOG: [ALERT] Animal identified in frame. Signal Sent.")

        except Exception as e:
            # If the drawing fails, the camera keeps running and just prints the error
            print(f"Drawing Error: {e}")

        current_time = time.time()
        if current_time - last_saved_time > save_cooldown:
            
            # Create a clean filename with date and time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_path}/hazard_{timestamp}.jpg"
            
            # Save the frame
            cv2.imwrite(filename, annotated_frame)
            
            # Update the timer
            last_saved_time = current_time
            print(f">>> EVIDENCE CAPTURED: {filename}")

    # 5. Show the result on your monitor
    cv2.imshow("PAWS - Predictive Animal Warning System", annotated_frame)

    cv2.imshow("PAWS - Predictive Animal Warning System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()