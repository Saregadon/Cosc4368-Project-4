#python3.7 -m pip install numpy scipy matplotlib pandas scikit-learn
#sudo apt-get install python3-tk inside of linux Terminal, not python environment
import csv
import sklearn as skl
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import linear_model
import matplotlib.pyplot as mp
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

#must read in entire csv file path
heartdisease = pd.read_csv("/home/useradd/Eick-4368-AI-Code/Cosc4368/Cosc4368-AI-Assignment-2/Cosc4368-Project-4/hearts.csv", sep = ',')
#print(inputfile.head())

print(heartdisease) #prints all data
print(heartdisease.head()) #prints first 5 sets #from pandas
print(heartdisease.info()) #prints the data types of our set

#print(data.isnull().sum()) #gives a summation of how many null values are in each one
#obviously for this, there will be 0 non-null values

label_quality = LabelEncoder()
heartdisease['high_blood_pressure'] = label_quality.fit_transform(heartdisease['high_blood_pressure'])

X = heartdisease.drop('high_blood_pressure', axis = 1)
y = heartdisease['high_blood_pressure']

#Train and Test splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

#Applying Linear Regression/Standard Scalar to get optimized result
#levels playing field in terms of extremely high numbers
sc = StandardScaler() #will also need to implement Linear Regression as well
X_train = sc.fit_transform(X_train, y_train)
X_test = sc.transform(X_test)

#print out variables
print(X_train[:10]) #shows the first 10 sets

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, sc, color = 'blue', linewidth = 3)

plt.xticks(())
plt.yticks(())

print("done")

#Concentration is to attempt to show the presence of heart disease in patients, starting from 0(absent) to 4 -- [0, 1, 2, 3, 4]
#possibly call data in 4 times
#have data used to run for Linear Regression, Decision Tree?
#then data used for Logistic/sigmoid function, Tanh, or Relu