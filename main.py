import cv2
import numpy as np

# Let's load the required XML classifiers.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

# Load the img in grey scale mode
img = cv2.imread("face.jpg")
grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# if face found, it return the position of detected faces
faces = face_cascade.detectMultiScale(grayScale, 1.2, 3)


for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
