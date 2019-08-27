import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix, andrews_curves, parallel_coordinates, radviz
import matplotlib.pyplot as plt


#data = pd.read_csv('test.csv')
#data = pd.read_csv('BostonHousing.csv')
#scatter_matrix(data, alpha=0.2, figsize=(10, 10), diagonal='kde')
#plt.suptitle('scatter-matrix')
#plt.show()

# data = pd.read_csv('BostonHousing.csv')
# data.plot.line(x='medv', y=['rad', 'indus', 'ptratio', 'lstat'], figsize=(10, 10))
# plt.show()

data = pd.read_csv('test.csv')
#data = data.cumsum()
radviz(data, 'medv')
plt.show()
# print (data.head())