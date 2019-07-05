#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pandas as pd
from sklearn.linear_model import LogisticRegression

# 模型效果衡量标准

data = pd.read_csv('Text/admissions.csv')
X = data.ix[:,1:].values
Y = data['admit']
lgr = LogisticRegression()
lgr.fit(X,Y)
data['admit_label'] = lgr.predict(X)

True_Positive = (data['admit_label'] ==1) & (data['admit'] ==1)
True_Positive = data[True_Positive]

True_Negatives = (data['admit_label'] == 0) & (data['admit'] == 0)
True_Negatives = data[True_Negatives]

False_Negatives = (data['admit_label'] ==0) & (data['admit'] ==1)
False_Negatives = data[False_Negatives]

False_Positive = (data['admit_label'] == 1) &(data['admit'] ==0 )
False_Positive = data[False_Positive]

print(len(True_Positive)/(len(False_Negatives) + len(True_Positive))) # 召回率
print(len(True_Positive)/(len(False_Positive) + len(True_Positive))) # 准确率

