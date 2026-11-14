import cv2
from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracking
from engine.utils import calculate_meters_per_pixel,  get_warp_matrix, warp_perspective
from engine.object_speed import ObjectSpeed
import numpy as np


od = ObjectDetection("models/vehicle.pt")


mot = MultiObjectTracking()
tracker = mot.oscort()

cap = cv2.VideoCapture("C:/Users/User/Desktop/pythonProject/object_tracking")


roi = np.array([
    (1330,130),
    (1568,191),
    (601,1045),
    (328,870)
], dtype= np.int32)


meter_per_pixel = calculate_meter_per_pixel((1330,130),(328, 870), real_distance_m=80)


warp_matrix = get_warp_matrix(roi)


fps = cap.get(cv2.CAP_PROP_FPS)

od_speed = ObjectSpeed(fps=fps, meter_per_pixel=meter_per_pixel, warp_matrix=warp_matrix)
print("Meters per pixel:", meter_per_pixel)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    bboxes, labels, score = od.detect(frame)

    bboxes_ids = tracker.update(bboxes,score, labels, frame)

    
    
    cv2.polylines(frame,[roi], isClosed=True, color=(0,255,0),thickness=2)
    
        
    for x1, y1, x2, y2, obj_id, class_id, score in np.array(bboxes_ids):
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
        cv2.putText(frame, f"V {obj_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

 
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if cv2.pointPolygonTest(roi, (cx,cy), False) >=0:
            speed = od_speed.update(obj_id,cx,cy)
            if speed > 0:
                speed_test = f"Speed: {speed:.2f} km/h"
                cv2.putText(frame, speed_test, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)



    roi_warped = warp_perspective(frame, warp_matrix, dst_size=(720,1280))

    cv2.imshow("Warped ROI",cv2.rotate(roi_warped, cv2.ROTATE_90_CLOCKWISE))
    cv2.imshow("Video Frame", frame)
    key= cv2.waitKey(1)

    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
