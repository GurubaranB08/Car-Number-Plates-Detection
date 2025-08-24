import cv2
import easyocr
import os

harcascade = "model/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) #height

min_area = 500
count = 0

os.makedirs("plates", exist_ok=True)
reader = easyocr.Reader(['en'])

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s') and img_roi is not None:
        img_filename = f"plates/scaned_img_{count}.jpg"
        txt_filename = f"plates/scaned_img_{count}.txt"
        cv2.imwrite(img_filename, img_roi)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results",img)
        cv2.waitKey(500)

        # --- EasyOCR step: analyze saved image and save text ---
        result = reader.readtext(img_filename)
        plate_text = ""
        if result:
            # Combine all detected text pieces for robustness
            plate_text = " ".join([res[1] for res in result]).strip()
        else:
            plate_text = "NOT_DETECTED"

        with open(txt_filename, "w") as f:
            f.write(plate_text)

        print(f"Saved image as {img_filename} and text as {txt_filename} (recognized: {plate_text})")

        count += 1
