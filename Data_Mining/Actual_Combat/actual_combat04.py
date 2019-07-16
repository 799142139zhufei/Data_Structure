#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


'''
对于多分类问题时如何解决：
遇到分类问题时常用的逻辑回归就不太适合比如：
有三个类别（A、B、C）假设三个场景
场景一：A、（B、C）2类计算A类的概率
场景二:（A、B）、C 2类计算C类的概率
场景三:（A、C）、B 2类计算B类的概率
当来一个测试数据后只需要预测在A、B、C类谁的概率最大即可
'''

columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'year', 'car name','origin']

data = pd.read_table('../Text/auto-mpg.data',names=columns,delim_whitespace=True)

# 特征变换独热编码
dummy_cylinders = pd.get_dummies(data['cylinders'],prefix='cly')
data = pd.concat([dummy_cylinders,data],axis=1)
data = data.drop('cylinders',axis=1)

dummy_year = pd.get_dummies(data['year'],prefix='year')
data = pd.concat([dummy_year,data],axis=1)
data = data.drop('year',axis=1)

features = []
for column in data.columns.tolist():
    if column.startswith('cly') or column.startswith('year'):
        features.append(column)

X = data.ix[:,features].values
Y = data['car name']

size = 0.33
random_state = 1
unique_carname = data['car name'].unique() # 对label进行分组
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,random_state=random_state,test_size= size)

models = {}
# 模型训练---根据label标签建立是三个模型
for carname in unique_carname:
    model = LogisticRegression()
    y_train = Y_train == carname
    model.fit(X_train,y_train)
    models[carname] = model


# 模型测试---根据测试数据进行预测
testing_probs = pd.DataFrame(columns=unique_carname)
for carname in unique_carname:
    testing_probs[carname] = models.get(carname).predict_proba(X_test)[:,1]
print(testing_probs)
