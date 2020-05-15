# Naive-Bayes

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Getting the Dataset
dataset = pd.read_csv('breast-cancer-wisconsin.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 10].values

# Handling Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:6])
X[:, 1:6] = imputer.transform(X[:, 1:6])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler #class
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix #function
cm = confusion_matrix(y_test, y_pred)