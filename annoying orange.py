import cv2
import dlib
from imutils import face_utils, resize
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True : 
    img = cv2.imread("face.jpg", cv2.IMREAD_COLOR)

    faces = detector(img)

    if len(faces) > 0:
        face = faces[0]

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy()

        shape = predictor(img, face)
        shape = face_utils.shape_to_np(shape)
        
        for p in shape:
            cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

        
        cv2.imshow('face', face_img)

    if cv2.waitKey(1) == ord('q'):
        break