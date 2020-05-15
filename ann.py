# Part 1 - Data Preprocessing

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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler #class
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the ANN!!!

# Importing the Keras libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the Input Layer and the First Hidden Layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(activation="relu", input_dim=9, units=9, kernel_initializer="uniform"))

# Adding Second Hidden Layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Hidden Layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding Output Layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting Classifier ANN to Dataset
classifier.fit(X_train, y_train, batch_size =180, epochs = 300)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix #function
cm = confusion_matrix(y_test, y_pred)
