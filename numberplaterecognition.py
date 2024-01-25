from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import easyocr

cap = cv2.VideoCapture(0)
cap.set(0, 384)
cap.set(0, 640)

model = YOLO("C:\\Users\\Joel\\Desktop\\Number plate\\num-pl-rec.pt")

classNames = ["numberplate"]

reader = easyocr.Reader(['en'])

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "numberplate" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)

                roi = img[y1:y2, x1:x2]

                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                result = reader.readtext(roi_rgb)

                text = result[0][1] if result else currentClass

                cv2.putText(img, f'{text}', (max(0, x1), max(35, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                currentArray = np.array([x1, y2, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    cv2.imshow("Result", img)
    cv2.waitKey(1)
