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

X = np.asarray(newdf)
y = np.asarray(df['DEATH_EVENT'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#only use serum creatinine and ejection fraction
#death event 1 == death, death event 0 == survival

#SVM linear work
svmlinear = SVC(kernel='linear',random_state = 1).fit(X_train, y_train).fit(X_train, y_train)

SVLCV = cross_validate(svmlinear, X_train, y_train, cv=10)
svlin = svmlinear.score(X_test, y_test)
print(SVLCV)
print(svlin)

#SVM poly work
svmpoly = SVC(kernel='poly',random_state = 1).fit(X_train, y_train).fit(X_train, y_train)

SVPCV = cross_validate(svmpoly, X_train, y_train, cv=10)
svmpol = svmpoly.score(X_test, y_test)
print(SVPCV)
print(svmpol)

#MLP logistic work
MLPlog = MLPClassifier(activation='logistic', random_state=1, max_iter=5000).fit(X_train, y_train)

logisticMLPcv = cross_validate(MLPlog, X_train, y_train, cv=10)
mlplogistics = MLPlog.score(X_test, y_test)
print(logisticMLPcv)
print(mlplogistics)

#MLP tanh work
MLPtan = MLPClassifier(activation='tanh', random_state=1, max_iter=5000).fit(X_train, y_train)

TanhMLPcv = cross_validate(MLPtan, X_train, y_train, cv=10)
fixmlp = mlplogistics.score(X_test, y_test)
print(TanhMLPcv)
print(fixmlp)

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