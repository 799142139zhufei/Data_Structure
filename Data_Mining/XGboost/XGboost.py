#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

'''
随机深林模型的基础模型和xgboost一样都是决策树模型定义数的个数、拆分的深度且是并行的，树与树之间不影响；
xgboost模型的基础模型也是决策树，区别在于它是串行的树与树之间相互影响，下一个树对前一个树进行优化提升；
'''

data = pd.read_csv('pima-indians-diabetes.csv',
                   names=['a1','a2','a3','a4','a5','a6','a7','label'])

X  = data.ix[:,0:7]
Y = data['label']
random_state = 7
test_size = 0.33
X_train, X_test, \
y_train, y_test = train_test_split(X,Y,
                                   test_size=test_size,
                                   random_state=random_state)
xgboost = XGBClassifier()
xgboost.fit(X_train.values,y_train)
test_predict = xgboost.predict(X_test.values)
score = accuracy_score(y_test,test_predict)
print('Accuracy_Score:'+ str(round(score*100,2))+ '%')


'''
# 梯度提升
data = pd.read_csv('pima-indians-diabetes.csv',
                   names=['a1','a2','a3','a4','a5','a6','a7','label'])
X = data.ix[:,0:7]
Y = data['label']

random_state = 7
test_size = 0.33
X_tarin,X_test,Y_tarin,Y_test= train_test_split(X,Y,test_size=test_size,random_state=random_state)
eval_set = [(X_test.values,Y_test)]
xgboost = XGBClassifier()
xgboost.fit(X_tarin.values,Y_tarin,early_stopping_rounds=10,
            eval_metric="logloss",eval_set=eval_set,verbose=True)
test_predict = xgboost.predict(X_test.values)
Accuracy_Score = accuracy_score(Y_test,test_predict)
print('Accuracy_Score：'+ str(round(Accuracy_Score*100,2)) + '%')
'''

data = pd.read_csv('pima-indians-diabetes.csv',
                   names=['a1','a2','a3','a4',
                          'a5','a6','a7','label'])
X = data.ix[:,0:7]
Y = data['label']
xgboost = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] # 不同的学习率
param_grid = dict(learning_rate = learning_rate)
random_state = 7

kfold = StratifiedKFold(n_splits=10,
                        random_state = random_state,
                        shuffle=True)

grid_search= GridSearchCV(xgboost,param_grid,
                          scoring = 'neg_log_loss',
                          n_jobs = -1 ,cv = kfold)

grid_result = grid_search.fit(X.values,Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print('---------------')
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f with: %r" % (mean, param))


#cnd2018知识库

