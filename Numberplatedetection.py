from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

#cap = cv2.VideoCapture("C:\\Users\\Joel\\Desktop\\Number plate\\Vid\\5.mp4") #For video
cap = cv2.VideoCapture(0)
cap.set(0, 384)
cap.set(0, 640)

model = YOLO("C:\\Users\\Joel\\Desktop\\Number plate\\num-pl-rec.pt")

classNames = ["numberplate"] #Pretrained objects

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
        #Bounding box
            x1, y1, x2, y2 = box.xyxy[0] 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            w, h = x2-x1, y2-y1

            #confidence 
            conf = math.ceil((box.conf[0]*100))/100

            #Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass =="numberplate" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1),max(35,y1)), scale=0.6, thickness=2, offset=3)
                currentArray = np.array([x1,y2,x2,y2,conf])
                detections = np.vstack((detections, currentArray))
            
                
    cv2.imshow("Result", img)
    cv2.waitKey(1)