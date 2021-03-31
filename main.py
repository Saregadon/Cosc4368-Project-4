#python3.7 -m pip install numpy scipy matplotlib pandas scikit-learn
#sudo apt-get install python3-tk inside of linux Terminal, not python environment
import csv
import sklearn as skl
import seaborn as sms
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as mp
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#must read in entire csv file path
data = pd.read_csv("/home/useradd/Eick-4368-AI-Code/Cosc4368/Cosc4368-AI-Assignment-2/Cosc4368-Project-4/hearts.csv", header = 0)
#print(inputfile.head())

print(data) #prints all data
print(data.head()) #prints first 5 sets #from pandas
print(data.info()) #prints the data types of our set



#print(data.isnull().sum()) #gives a summation of how many null values are in each one
#obviously for this, there will be 0 non-null values

print("done")

#Concentration is to attempt to show the presence of heart disease in patients, starting from 0(absent) to 4 -- [0, 1, 2, 3, 4]
#possibly call data in 4 times
#have data used to run for Linear Regression, Decision Tree?
#then data used for Logistic/sigmoid function, Tanh, or Relu