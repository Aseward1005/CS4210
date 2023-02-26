#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#loop your data to allow each instance to be your test set
wrong = 0 #for calculating error rate
for i in range(len(db)):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #also adding classes in the same loop
    #--> add your Python code here
    X = []
    Y = []
    for j, row in enumerate(db):
        if not j == i: #for all data that is not the current instance
            temp = []
            for k in range(len(row)-1):
                temp.append(float(row[k]))
            X.append(temp)
            Y.append(row[len(row)-1])

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    translations = {'-': 0.0,
                    '+': 1.0}
    Y = [translations[i] for i in Y]
 

    

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = db[i][0:len(db[i])-1]
    testSample = [float(i) for i in testSample]
    

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if (not class_predicted == translations[db[i][len(db[i])-1]]): #if the predicted class is not the same as the numeric value for the actual class
        wrong += 1

#print the error rate
#--> add your Python code here
print(f"error = {wrong/len(db)}")






