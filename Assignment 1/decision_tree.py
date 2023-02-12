#-------------------------------------------------------------------------
# AUTHOR: Anthony Seward
# FILENAME: decision_tree.py
# SPECIFICATION: implementation of a decision tree
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
#i could automate this in the for loop, but it seemed quicker to do manually for this one
#define the numeric translations in a dictionary
translations = {'young': 0, 'prepresbyopic': 1, 'presbyopic': 2, 
                'myope': 0, 'hypermetrope': 1,
                'yes': 0, 'no': 1,
                'reduced': 0, 'normal': 1}
rowtrans = [0, 0, 0, 0]
for i in range(len(db)): #all rows
  for j in range(len(db[i]) - 1): #all columns except the last
    rowtrans[j] = translations[db[i][j].lower()] #translate all the strings to numbers in each row
  X.append(tuple(rowtrans))

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
for i in range(len(db)): 
  Y.append(translations[db[i][len(db[i])-1].lower()]) #add the translation of the last column in the row

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()