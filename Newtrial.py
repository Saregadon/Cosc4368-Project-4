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
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy as sp
import pandas as pd
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

svmaccuracies = {}
lraccuracies = {}

ifs = "/home/useradd/Eick-4368-AI-Code/Cosc4368/Cosc4368-AI-Assignment-2/Cosc4368-Project-4/hearts.csv"

#must read in entire csv file path
df = pd.read_csv(ifs, header=0)
df.corr()

newdf = df[['ejection_fraction', 'serum_creatinine']]

x = np.asarray(newdf)
y = np.asarray(df['DEATH_EVENT'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

#only use serum creatinine and ejection fraction
#death event 1 == death, death event 0 == survival

#SVM rbf
svmrbf = SVC(kernel='rbf',random_state = 1).fit(x_train, y_train)
SVRCV = cross_validate(svmrbf, x_train, y_train, cv=10)

#SVM sigmoid
svmsigmoid = SVC(kernel='sigmoid',random_state = 1).fit(x_train, y_train)
SVSCV = cross_validate(svmsigmoid, x_train, y_train, cv=10)

#MLP tanh
MLPtanh = MLPClassifier(activation='tanh', random_state=1, max_iter=3000).fit(x_train, y_train)
tanMLPcv = cross_validate(MLPtanh, x_train, y_train, cv=10)

#MLP relu
MLPrelu = MLPClassifier(activation='relu', random_state=1, max_iter=3000).fit(x_train, y_train)
ReluMLPcv = cross_validate(MLPrelu, x_train, y_train, cv=10)

svrbf = svmrbf.score(x_test, y_test)
svmsig = svmsigmoid.score(x_test, y_test)
Tanh = MLPtanh.score(x_test, y_test)
fixmlp = MLPrelu.score(x_test, y_test)

print("Rbf kernel accuracy is {}%".format(svrbf, svmsig))
print("Sigmoid kernel accuracy is {}%".format(svmsig))
print("tanh MPL accuracy is {}%".format(Tanh))
print("Relu MPL accuracy is {}%".format(fixmlp))

plt.scatter(x=df.ejection_fraction[df.DEATH_EVENT==1], y=df.serum_creatinine[(df.DEATH_EVENT==1)], c="red")
plt.scatter(x=df.ejection_fraction[df.DEATH_EVENT==0], y=df.serum_creatinine[(df.DEATH_EVENT==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Ejection Fraction")
plt.ylabel("Serum Creatinine")
plt.show()

print("done")

#Concentration is to attempt to show the presence of heart disease in patients, starting from 0(absent) to 4 -- [0, 1, 2, 3, 4]
#possibly call data in 4 times
#have data used to run for Linear Regression, Decision Tree?
#then data used for Logistic/sigmoid function, Tanh, or Relu