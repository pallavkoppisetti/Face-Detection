import cv2
import os
import numpy as np

def load_train_data(num=50):
    # open directory
    faces = os.listdir('train/face')

    ind = np.random.choice(len(faces), num, replace=False)

    images = []
    labels = []

    for i in ind:
        image = cv2.imread('train/face/' + faces[i], -1)
        images.append(image)
        labels.append(1)

    non_faces = os.listdir('train/non-face')

    for i in ind:
        image = cv2.imread('train/non-face/' + non_faces[i], -1)
        images.append(image)
        labels.append(0)

    return np.array(images), np.array(labels)

def load_test_data(num=50):

    faces = os.listdir('test/face')

    images = []
    labels = []

    ind = np.random.choice(len(faces), num, replace=False)

    for i in ind:
        image = cv2.imread('test/face/' + faces[i], -1)
        images.append(image)
        labels.append(1)

    non_faces = os.listdir('test/non-face')

    for i in ind:
        image = cv2.imread('test/non-face/' + non_faces[i], -1)
        images.append(image)
        labels.append(0)

    return np.array(images), np.array(labels)


if __name__=="__main__":
    images, labels = load_train_data()
    print(images.shape)