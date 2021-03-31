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
from sklearn.metrics import classification_report
from sklearn import linear_model
import numpy as np
import scipy as sp
import pandas as pd
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

#print(os.listdir("/home/useradd/Eick-4368-AI-Code/Cosc4368/Cosc4368-AI-Assignment-2/Cosc4368-Project-4"))

ifs = "/home/useradd/Eick-4368-AI-Code/Cosc4368/Cosc4368-AI-Assignment-2/Cosc4368-Project-4/hearts.csv"

#must read in entire csv file path
data = pd.read_csv(ifs, header=0)
print(data.head(5))

#ASSIGN the data
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#SPLIT to test and train the data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

#NORMALIZE the massive numbers to percentages so everything is equally weighted
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

modelLG = LogisticRegression(random_state=1) #instance of model
modelLG.fit(x_train, y_train) #train/fit model

y_predLG = modelLG.predict(x_test) #gets y predictions
print(classification_report(y_test, y_predLG))

plt.plot(classification_report(y_test, y_predLG))
plt.xlabel('heart disease')
plt.ylabel('age')
plt.show()

#summarizes the count, the mean, the standard deviation, the min and the max numeric variables.
print(data.describe())

#prints data values that are null || 0
#print(data.isnull().sum())

#calculate correlation matrix
corr = data.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

#subData = data[['age', 'trestbps','chol','thalach','oldpeak']]
#sns.pairplot(subData)

print("done")

#Concentration is to attempt to show the presence of heart disease in patients, starting from 0(absent) to 4 -- [0, 1, 2, 3, 4]
#possibly call data in 4 times
#have data used to run for Linear Regression, Decision Tree?
#then data used for Logistic/sigmoid function, Tanh, or Relu