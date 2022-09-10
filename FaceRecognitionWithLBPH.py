from PIL import Image
import cv2
import numpy as np
import zipfile
import os
from numpy import asarray
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn

#path = 'Datasets/yalefaces.zip'
#zip_object = zipfile.ZipFile(file=path, mode='r')
#zip_object.extractall('./')
#zip_object.close()

# Preprocessing the image
print(os.listdir('yalefaces/train'))


def get_image_detail():
    paths = [os.path.join('yalefaces/train', f) for f in os.listdir('yalefaces/train')]
    print(paths)
    faces1 = []
    ids1 = []
    for path1 in paths:
        image = Image.open(path1).convert('L')
        # image_np = np.array(image, 'uint8')
        # image_np = asarray(image)
        # image_np = cv2.imread(image, mode='')
        image_np = imread(path1)
        id1 = int(os.path.split(path1)[1].split('.')[0].replace('subject', ''))
        # print(id1)
        ids1.append(id1)
        faces1.append(image_np)

    return np.array(ids1), faces1


ids, faces = get_image_detail()
print(len(ids))
print(len(faces))
print(faces[0], faces[0].shape)

# Training the LBPH classifier

lbph_classifier = cv2.face_LBPHFaceRecognizer.create(radius=4, neighbors=14, grid_x=9, grid_y=9)
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')

# Recognizing faces

lbph_face_classifier = cv2.face_LBPHFaceRecognizer.create()
lbph_face_classifier.read('lbph_classifier.yml')

test_image = 'yalefaces/test/subject10.sad.gif'
test_image_np= imread(test_image)
print(test_image_np)
print(test_image_np.shape)
prediction = lbph_face_classifier.predict(test_image_np)
print(prediction)

expected_output  = int(os.path.split(test_image)[1].split('.')[0].replace('subject', ''))
print(expected_output)
cv2.putText(test_image_np, 'Pred:' + str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.putText(test_image_np, 'Exp:' + str(expected_output), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.imshow('Face Detection:' + str(prediction[0]), test_image_np)
#cv2.waitKey(0)

# Evaluating the face classifier
paths_test = [os.path.join('yalefaces/test', f) for f in os.listdir('yalefaces/test')]
predictions = []
expected_outputs = []
for path in paths_test:
    image_np = imread(path)
    pred, _ = lbph_face_classifier.predict(image_np)
    exp = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

    predictions.append(pred)
    expected_outputs.append(exp)

predictions = np.array(predictions)
expected_outputs = np.array(expected_outputs)

print(predictions)
print(expected_outputs)

print(accuracy_score(expected_outputs, predictions))
print(confusion_matrix(expected_outputs, predictions))
seaborn.heatmap(confusion_matrix(expected_outputs, predictions), annot= True)
plt.show()



