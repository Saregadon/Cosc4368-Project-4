#python3.7 -m pip install numpy scipy matplotlib pandas scikit-learn
#sudo apt-get install python3-tk inside of linux Terminal, not python environment
import csv
import sklearn as skl
from sklearn import svm, preprocessing
from sklearn import datasets
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy as sp
import pandas as pd
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

#initialize
def initialize(dimension):
    
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias

accuracies = {}

#print(os.listdir("/home/useradd/Eick-4368-AI-Code/Cosc4368/Cosc4368-AI-Assignment-2/Cosc4368-Project-4"))

ifs = "/home/useradd/Eick-4368-AI-Code/Cosc4368/Cosc4368-AI-Assignment-2/Cosc4368-Project-4/hearts.csv"

#must read in entire csv file path
df = pd.read_csv(ifs, header=0)
print(df.corr())
#print(df)
#print(df.head(5))

#plt.figure(figsize=(7,7))
#plt.show()
#sns.heatmap(df.corr(), annot=True, fmt='.0%')

#remove Death_event cell
#df = df.drop(['DEATH_EVENT'], axis = 1)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

sc = StandardScaler(df)

#only use serum creatinine and ejection fraction
#death event 1 == death, death event 0 == survival

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=1)
forest.fit(x_train, y_train)

model = forest
model.score(x_train, y_train)

lr = LogisticRegression(y_test, model.predict(x_test))

TN = lr[0][0]
TP = lr[1][1]
FN = lr[1][0]
FP = lr[0][1]

print("Test Accuracy {}".format((TP+TN)/ (TP + TN + FN + FP)))

"""
y = df.DEATH_EVENT.values
x_data = df.drop(['DEATH_EVENT'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values #normalization

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

predicted = cross_val_predict(estimator='predict', X=x_train, y=y_train, cv=10, n_jobs=-1, fit_params=None)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
"""
print("done")

#Concentration is to attempt to show the presence of heart disease in patients, starting from 0(absent) to 4 -- [0, 1, 2, 3, 4]
#possibly call data in 4 times
#have data used to run for Linear Regression, Decision Tree?
#then data used for Logistic/sigmoid function, Tanh, or Relu