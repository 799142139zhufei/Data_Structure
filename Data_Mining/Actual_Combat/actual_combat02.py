#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression

# 逻辑回归模型--预测是否能被大学录取

data = pd.read_csv('../Text/admissions.csv')
#plt.scatter(data['gpa'],data['admit']) # 参数一是X轴；参数二是Y轴
lir = LinearRegression()
X = data.ix[:,1:]
Y = data['admit']
lir.fit(X.values,Y)
lir_predict = lir.predict(X.values)
lir_predict_score = lir.score(X.values,Y)
print('LinearRegression线性回归模型预测分值：' + str(lir_predict_score))

lgr = LogisticRegression()
lgr.fit(X.values,Y)
lgr_predict = lgr.predict(X.values)
lgr_predict_score = lir.score(X.values,Y)
print('LogisticRegression逻辑回归预测分值：'+ str(lgr_predict_score))