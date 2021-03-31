#python3.7 -m pip install numpy scipy matplotlib pandas scikit-learn
#sudo apt-get install python3-tk inside of linux Terminal, not python environment
import csv
import sklearn as skl
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
#from sklearn.externals import joblib
import matplotlib.pyplot as mp
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#must read in entire csv file path
inputfile = pd.read_csv("/home/useradd/Eick-4368-AI-Code/Cosc4368/Cosc4368-AI-Assignment-2/Cosc4368-Project-4/hearts.csv", header = 0)
#print(inputfile.head())

print(inputfile)

"""
dataset = pd.DataFrame()

lr = linear_model.LinearRegression()
SVM = inputfile
y = SVM.target

#predicted has cross validation of 10 here -- cv == 10
predicted = cross_val_predict(lr, SVM.data, y, cv = 10)

fig, ax = plt.subplots()
ax.scatter(y,predicted)

#
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

ax.set_xlabel('Failure Rates')
ax.set_ylabel('Heart Rates')
plt.show()
"""

print("done")