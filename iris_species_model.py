import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/iris_species.csv")

y = df["Species"].to_numpy()

x = df.drop(columns="Species").to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

model = LogisticRegression(max_iter = 1000)
model.fit(x_train_scaled,y_train)

y_predict = model.predict(x_test_scaled)

print(f"accuracy with scaling {metrics.accuracy_score(y_test,y_predict)}")

cf_matrix = metrics.confusion_matrix(y_test,y_predict)

print(cf_matrix)

