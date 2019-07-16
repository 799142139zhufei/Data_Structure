#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 线性回归模型----评估一加仑汽油的行驶多少公里
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model year', 'origin', 'car name']

data = pd.read_table('../Text/auto-mpg.data',names=columns,delim_whitespace=True)

fig = plt.figure(figsize=(3,4))
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
data.plot('weight', 'mpg', kind='scatter', ax=ax1)
data.plot('acceleration', 'mpg', kind='scatter', ax=ax2)
plt.show()

LR = LinearRegression()
X = data.ix[:,0:2]
Y = data['weight']
LR.fit(X.values,Y)
test_predict =  LR.predict(X.values)
mean = mean_absolute_error(Y,test_predict)
print(round(mean*0.5,2))