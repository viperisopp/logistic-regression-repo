import cv2 as cv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

path = "/Users/simon/.cache/kagglehub/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda/versions/1"

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

cat_data, cat_target = load_images_from_folder(path+"/animals/cats/","cats")
dog_data, dog_target = load_images_from_folder(path+"/animals/dogs/","dogs")
panda_data, panda_target = load_images_from_folder(path+"/animals/panda/","pandas")

x = cat_data + dog_data + panda_data
y = cat_target + dog_target + panda_target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

x_train = np.array(x_train,dtype=np.float32) / 255.0
y_train = np.array(y_train)
x_test = np.array(x_test,dtype=np.float32) / 255.0
y_test = np.array(y_test)

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

model = LogisticRegression(class_weight="balanced",max_iter=10000)
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(f"model accuracy score: {metrics.accuracy_score(y_test,y_predict)}")
print(metrics.confusion_matrix(y_test,y_predict))