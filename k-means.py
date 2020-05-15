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

#Encoding the Y set
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
'''wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter= 700, n_init= 40 , random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1 , 11) , wcss)
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()'''

# Applying K-Means To Dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter= 700, n_init= 40 , random_state = 0)
y_kmeans = kmeans.fit_predict(X)

'''# Visualising the Clusters
plt.scatter(X[y_kmeans == 0, :], y[y_kmeans == 0, 0], s = 100, c = 'red', label = 'B')
plt.scatter(X[y_kmeans == 1, :], y[y_kmeans == 1, 0], s = 100, c = 'blue', label = 'M')
plt.show()'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix #function
cm = confusion_matrix(y, y_kmeans)