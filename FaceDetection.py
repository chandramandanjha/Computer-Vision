import cv2

image = cv2.imread('Images/people1.jpg')
print(image.shape)

#image = cv2.resize(image, (800, 600))
#print(cv2.imshow('Test', image))

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image_gray.shape)
#print(cv2.imshow('Test', image_gray))
#cv2.waitKey(0)  #for pycharm

# Detecting Faces
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
face_detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.3, minSize=(30,30))

print(face_detections)
print(len(face_detections))  # No of detections
for(x, y, w, h) in face_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
print(cv2.imshow('Test', image))

eye_detections = eye_detector.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, maxSize=(70,70))
for(x, y, w, h) in eye_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
print(cv2.imshow('Test', image))
cv2.waitKey(0)  #for pycharm

image2 = cv2.imread('Images/people2.jpg')
image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
detections2 = face_detector.detectMultiScale(image_gray2, scaleFactor = 1.2, minNeighbors=7, minSize=(10,10), maxSize=(100,100))
for(x, y, w, h) in detections2:
    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
print(cv2.imshow('Test', image2))
#cv2.waitKey(0)  #for pycharm
