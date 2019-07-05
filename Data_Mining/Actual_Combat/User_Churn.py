#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

'''
业务场景描述：预测某网站用户流失情况
'''

# 1、数据预处理
data = pd.read_csv('Text/churn.csv')
#columns = data.columns.tolist() 获取所有的列
churn_result = data['Churn?'] # 标签列

y = np.where(churn_result == 'True.',1,0) # True（流失） 、False（未流失）
#print(len(y[y == 1]),len(y[y==0])) 正负样本不均（向上采样或者向下采样）

drop_row = ['State','Area Code','Phone','Churn?'] # 根据业务场景删除一些无相关的特征，避免影响模型的效果
data = data.drop(drop_row,axis =1) # 基于列删除

# 将Int'l Plan、VMail Plan进行特征变换转换成0、1
yes_no_cols = data[["Int'l Plan","VMail Plan"]]
data[["Int'l Plan","VMail Plan"]] = np.where(yes_no_cols == 'yes',1,0)

x = data.as_matrix().astype(np.float) # 数据变换---后面进行数值归一化或数据的标准化
# mms = MinMaxScaler()  数值归一化（0到1）
# x = mms.fit_transform(x)
ss = StandardScaler() # 数值标准化（-1到1）
x = ss.fit_transform(x)

#2、交叉验证数据集----通过多个模型得出对应的精确率

def run_cv(x,y,clf_model,**kwargs):
    KFoldS = KFold(n_splits= 5,random_state =1,shuffle=True)
    y_pred = y.copy()
    for train_c,test_c in KFoldS.split(x):
        train_x,train_y = x[train_c],y[train_c]
        test_x = x[test_c]
        clf = clf_model(**kwargs)
        clf.fit(train_x,train_y)
        y_pred[test_c] = clf.predict(test_x)
        #print(y_pred,len(y_pred),len(y_pred[y_pred ==1]),len(y_pred[y_pred ==0]))
    return y_pred

def accuracy(y_true,y_predict):
    return  np.mean(y_true == y_predict)

print ("Support vector machines:")
print ("%.3f" % accuracy(y, run_cv(x,y,SVC)))
print ("Random forest:")
print ("%.3f" % accuracy(y, run_cv(x,y,RandomForestClassifier)))
print ("K-nearest-neighbors:")
print ("%.3f" % accuracy(y, run_cv(x,y,KNeighborsClassifier)))

#3、由于精确率不能真实的反映一个模型的好坏需要通过精确率和召回率同时反映

def run_prob_cv(x,y,clf_model,**kwargs):

    KFold1 = KFold(n_splits=5, random_state=1, shuffle=True)
    y_pred_prob = np.zeros((len(y),2))
    for train_c, test_c in KFold1.split(x):
        train_x, train_y = x[train_c], y[train_c]
        test_x = x[test_c]
        clf = clf_model(**kwargs)
        clf.fit(train_x, train_y)
        y_pred_prob[test_c] = clf.predict_proba(test_x)
    return y_pred_prob

pred_prob = run_prob_cv(x, y, RandomForestClassifier, n_estimators=10)
pred_churn = pred_prob[:,1] # 预测label为1的概率
is_churn = y == 1 # 真实label为1
counts = pd.value_counts(pred_churn)
true_prob = {}
# 统计真实label的均值
for i in counts.index:
    true_prob[i] = np.mean(is_churn[pred_churn == i])
    true_prob = pd.Series(true_prob)

# 对比预测概率为多少时有多个人员会流失实际流失的概率又是多少
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']

print(counts)