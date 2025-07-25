import cv2 as cv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# due to file size, I have not pushed the cat and dog image files 
# you can find the dataset I used at https://www.kaggle.com/tongpython/cat-and-dog!


def load_images_from_folder(folder,label):
    training_data = []
    training_target = []
    for image in os.listdir(folder):
        path = os.path.join(folder,image)
        img = cv.imread(path,cv.IMREAD_GRAYSCALE)
        img = cv.resize(img,(128,128))
        training_data.append(img)
        training_target.append(label)
    return training_data, training_target

cat_training, cat_training_target = load_images_from_folder("../catdog/training_set/cats/","cats")
dog_training, dog_training_target = load_images_from_folder("../catdog/training_set/dogs/","dogs")

training_data = cat_training + dog_training 
training_target = cat_training_target + dog_training_target

################# visual separator #####################

cat_testing, cat_testing_target = load_images_from_folder("../catdog/test_set/cats/","cats")
dog_testing, dog_testing_target = load_images_from_folder("../catdog/test_set/dogs/","dogs")

testing_data = cat_testing + dog_testing 
testing_target = cat_testing_target + dog_testing_target

# convert to numpy array to normalize and reshape

training_data = np.array(training_data,dtype=np.float32) / 255.0
training_target = np.array(training_target)

testing_data = np.array(testing_data,dtype=np.float32) / 255.0
testing_target = np.array(testing_target)

training_data = training_data.reshape(len(training_data),-1)
testing_data = testing_data.reshape(len(testing_data),-1)

# fit and train model

model = LogisticRegression(max_iter=10000)
model.fit(training_data,training_target)

# predict and assess accuracy

testing_predict = model.predict(testing_data)
print(f"model accuracy score: {metrics.accuracy_score(testing_target,testing_predict)}")
print(metrics.confusion_matrix(testing_target,testing_predict))