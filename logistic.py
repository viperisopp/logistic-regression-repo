import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("datasets/indian_diabetes.csv")
print(df)