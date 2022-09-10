import dlib
import cv2

image = cv2.imread('Images/people2.jpg')
face_detector_hog = dlib.get_frontal_face_detector()
detections = face_detector_hog(image, 1)
print(detections)

for face in detections:
    # print(face.left())
    # print(face.top())
    # print(face.right())
    # print(face.bottom())
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)
cv2.imshow('People', image)
cv2.waitKey(0)  # for pycharm