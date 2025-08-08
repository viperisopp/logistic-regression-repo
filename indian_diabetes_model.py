import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/indian_diabetes.csv")

y = df["Outcome"].to_numpy()
x = df.drop(columns="Outcome").to_numpy()
x = np.reshape(x,(-1,8))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model_without_preprocessing = LogisticRegression(max_iter = 1000)
model_without_preprocessing.fit(x_train,y_train)

y_predict = model_without_preprocessing.predict(x_test)

print(f"accuracy without scaling {metrics.accuracy_score(y_test,y_predict)}")

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

model_with_preprocessing = LogisticRegression(max_iter = 1000)
model_with_preprocessing.fit(x_train_scaled,y_train)

y_predict = model_with_preprocessing.predict(x_test_scaled)

print(f"accuracy with scaling {metrics.accuracy_score(y_test,y_predict)}")

cf_matrix = metrics.confusion_matrix(y_test,y_predict)

print(cf_matrix)

true_positive = cf_matrix[0][0]
true_negative = cf_matrix[1][1]
false_positive = cf_matrix[1][0]
false_negative = cf_matrix[0][1]


'''
Takeaways:
- I learned about many different statistical definitions like:
    + Standard deviation
    + variance
    + normalization
    + regularization
    + gaussian distribution (just normal distribution)
    + features
- gained knowledge on how logistic regression works, with many differnt algorithms to get models
    + lbfgs
    + saga
    + adam
    + newton
- Learned how to scale data
- scaling is important to reduce bias for features, scaling methods include:
    + stanardization
    + normalization
    + min-max scaling
- familiarized myself with more scikit-learn functions and sublibraries
ts is so linkedin pilled
'''