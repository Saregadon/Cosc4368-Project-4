#python3.7 -m pip install numpy scipy matplotlib pandas scikit-learn
#sudo apt-get install python3-tk inside of linux Terminal, not python environment
import sklearn as skl
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as mp
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sklearn as skl

ifs = "hearts.csv"

inputfile = pd.read_csv(ifs, header = 0)

dataset = pd.DataFrame()

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

#predicted has cross validation of 10 here -- cv == 10
predicted = cross_val_predict(lr, boston.data, y, cv = 10)

fig, ax = plt.subplots()
ax.scatter(y,predicted)

#
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

print("done")