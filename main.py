"""
CS619 - Test Phase

Run on a single image
This command runs the model on a single image, and output the image with face detected.

> python main.py --image path_to_image_file

"""

import cv2
import argparse


def opencvFaceDetection(image):

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # Read the image
    img = cv2.imread(image)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the resulting image
    cv2.imshow('Face Detection - Test Phase', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to image file")
    args = parser.parse_args()

    opencvFaceDetection(args.image)
