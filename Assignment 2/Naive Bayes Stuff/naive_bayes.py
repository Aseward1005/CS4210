#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#define the training and test files
training_file = 'weather_training.csv'
test_file = 'weather_test.csv'

#reading the training data in a csv file
#--> add your Python code here
db = []
with open(training_file) as train:
    reader = csv.reader(train)
    for i, row in enumerate(reader):
        if i > 0: #skip the header
            db.append(row)


#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#and
#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]

#--> add your Python code here
translations = {'sunny': 0, 'overcast': 1, 'rain': 2,
                'hot': 0, 'mild': 1, 'cool': 2,
                'normal': 0, 'high': 1,
                'weak': 0, 'strong': 1,
                'no': 0, 'yes': 1}

X = []
Y = []
for row in db:
    temp = [translations[row[i].lower()] for i in range(1, len(row)-1)]
    X.append(temp)
    Y.append(translations[row[len(row)-1].lower()])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
testdb = []
with open(test_file) as test:
    reader = csv.reader(test)
    for i, row in enumerate(reader):
        testdb.append(row)

#printing the header os the solution
#--> add your Python code here
for name in testdb.pop(0):
    print(name, end = '\t')

print('Confidence')

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
ans = ['No', 'Yes']
for row in testdb:
    temp = [translations[row[i].lower()] for i in range(1, len(row)-1)]
    predictions = clf.predict_proba([temp])
    for i, prediction in enumerate(predictions[0]):
        if (prediction > .75):
            for j in range(len(row)-1):
                if row[j].lower() == "overcast":
                    row[j] = "overcst"
                print(row[j], end = '\t')
                if j == 2 or j == 3:
                    print('', end = '\t')
    
            print(ans[i], end = '\t\t')
            print(prediction)
        

