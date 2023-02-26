#-------------------------------------------------------------------------
# AUTHOR: Anthony Seward
# FILENAME: decision_tree_2.py
# SPECIFICATION: 
# FOR: CS 4210- Assignment #2
# TIME SPENT: about 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
testSet = 'contact_lens_test.csv'

translations = {'young': 0, 'prepresbyopic': 1, 'presbyopic': 2, 
                'myope': 0, 'hypermetrope': 1,
                'yes': 0, 'no': 1,
                'reduced': 0, 'normal': 1}

# read the test data and add this data to dbTest
dbTest = []
# import the test set
with open(testSet, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skip the header
            dbTest.append(row)

# transform the features of the test instances to numbers (only needs to happen once, before the loop)
dbTestTrans = []
rowtrans = [0, 0, 0, 0, 0]
for i in range(len(dbTest)): #all rows
    for j in range(len(dbTest[i])): #all columns
        rowtrans[j] = translations[dbTest[i][j].lower()] #translate all the strings to numbers in each row
    dbTestTrans.append(tuple(rowtrans)) #lists are immutable, tuples are not

#save the accuracies in a list. each accuracy corresponds to one run of the loop
accs = []
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append(row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # partially done before the loop as a small optimization
    
    rowtrans = [0, 0, 0, 0]
    for i in range(len(dbTraining)): #all rows
        for j in range(len(dbTraining[i]) - 1): #all columns except the last
            rowtrans[j] = translations[dbTraining[i][j].lower()] #translate all the strings to numbers in each row
        X.append(tuple(rowtrans)) #lists are immutable, tuples are not

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for i in range(len(dbTraining)): 
        Y.append(translations[dbTraining[i][len(dbTraining[i])-1].lower()]) #add the translation of the last column in the row

    #loop your training and test tasks 10 times here
    #so we have the average accuracy of all 10 runs, rather than the accuracy of one run
    amount = 0
    correct = 0
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #for debugging/old way of doing this
        #predictions = []
        #dbTestTrue = []
        for data in dbTestTrans:
            amount += 1 #keep a running counter of how many instances there are

            #use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here

            classPrediction = clf.predict([data[0:(len(data)-1)]])[0]
            trueValue = data[4] 
            #old way of doing this section
            #predictions.append(classPrediction)
            #dbTestTrue.append(trueValue) #only needs to happen once but this is less lines of code

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if (classPrediction == trueValue):
                correct += 1
        #for debugging
        #print(predictions)
        #print(dbTestTrue)
    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    accs.append(correct/amount)
#print the average accuracy of this model during the 10 runs (training and test set).
#your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
#--> add your Python code here
for i in range(len(accs)):
    print(f'The accuracy when training on {dataSets[i]} is {accs[i]}')