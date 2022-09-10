import cv2

# Car Detections
image = cv2.imread('Images/car.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

car_detector = cv2.CascadeClassifier('Cascades/cars.xml')
car_detections = car_detector.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=5)
for(x, y, w, h) in car_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
print(cv2.imshow('Cars', image))
# cv2.waitKey(0)  # for pycharm

# Clock Detections
image2 = cv2.imread('Images/clock.jpg')
image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

clock_detector = cv2.CascadeClassifier('Cascades/clocks.xml')
clock_detections = clock_detector.detectMultiScale(image_gray2, scaleFactor=1.03, minNeighbors=1)
for(x, y, w, h) in clock_detections:
    cv2.rectangle(image2, (x, y), (x + w, y + h), (255, 0, 0), 2)
#print(cv2.imshow('Clocks', image2))
# cv2.waitKey(0)  # for pycharm

# Body Detections
image3 = cv2.imread('Images/people3.jpg')
image_gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

body_detector = cv2.CascadeClassifier('Cascades/fullbody.xml')
body_detections = body_detector.detectMultiScale(image_gray3, scaleFactor=1.05, minNeighbors=5, minSize=(50,50))
for(x, y, w, h) in body_detections:
    cv2.rectangle(image3, (x, y), (x + w, y + h), (255, 0, 0), 2)
print(cv2.imshow('Full Body', image3))
cv2.waitKey(0)  # for pycharm
