#-------------------------------------------------------------------------
# AUTHOR: Anthony Seward
# FILENAME: clustering.py
# SPECIFICATION: performs k-means clustering
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library
X_training = df.values

#assign your training data to X_training feature matrix

sc = 0 #silhouette coefficient
maxk = 0

#run kmeans testing different k values from 2 until 20 clusters
for k in range(2, 21):
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
     kmeans = KMeans(n_clusters=k, random_state=0, n_init=10) #set n_init to suppress the warning
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     newsil = silhouette_score(X_training, kmeans.labels_)
     if (sc < newsil):
          sc = newsil
          maxk = k
     plt.plot(k, newsil, 'bo')
     #print(f'{k}, {newsil}')

print(maxk)

     


#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df =  pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df.values).reshape(1,df.shape[0])[0]
#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
