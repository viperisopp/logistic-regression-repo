import cv2 as cv
import os
import numpy

# due to file size, I have not pushed the cat and dog image files 
# you can find the dataset I used at https://www.kaggle.com/tongpython/cat-and-dog!

training_data = []
training_target = []
for cat_image in os.listdir("../training_set/cats/"):
    path = os.path.join("../training_set/cats/",cat_image)
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(128,128))
    training_data.append(img)
    training_target.append(0)

for dog_image in os.listdir("../training_set/dogs/"):
    path = os.path.join("../training_set/dogs/",dog_image)
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(128,128))
    training_data.append(img)
    training_target.append(0)

testing_data = []
testing_target = []
for cat_image in os.listdir("../test_set/cats/"):
    path = os.path.join("../test_set/cats/",cat_image)
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(128,128))
    testing_data.append(img)
    testing_target.append(0)

for dog_image in os.listdir("../test_set/dogs/"):
    path = os.path.join("../test_set/dogs/",dog_image)
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(128,128))
    testing_data.append(img)
    testing_target.append(0)

