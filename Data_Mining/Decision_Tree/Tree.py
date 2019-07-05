#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import numpy as np
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier


'''
决策树选择信息增益率最大的作为树节点；
随机森林：随机指的是数据、特征的随机（数据量、特征要一样多）；森林指的是由多个决策数组成。

特征代表性可以通过信息熵增益来衡量，信息熵增幅越大特征越具有代表性；
loc——通过行标签索引行数据  loc['a'] 具体的行名称
iloc——通过行号索引行数据    iloc[0] 具体的行号
ix——通过行标签或者行号索引行数据（基于loc和iloc 的混合）

基于现有的船员数据预测是否能被获救
'''

######################训练集数据变换##############################
data_train = pd.read_csv('titanic_train.csv')
# print(data.describe()) # 生成简要的统计信息
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())  # 以均值的形式填充缺失值

# 性别转换
data_train.ix[data_train['Sex'] == 'male', 'Sex'] = 0
data_train.ix[data_train['Sex'] == 'female', 'Sex'] = 1

# 上传地点转换
# data1 = data.groupby('Embarked').size() 由于数据中S占大多数所以将NaN赋值为S
data_train['Embarked'] = data_train['Embarked'].fillna('S')
data_train.ix[data_train['Embarked'] == 'S', 'Embarked'] = 0
data_train.ix[data_train['Embarked'] == 'C', 'Embarked'] = 1
data_train.ix[data_train['Embarked'] == 'Q', 'Embarked'] = 2

###########################测试集数据变换############################
data_test = pd.read_csv('test.csv')
# print(data.describe()) # 生成简要的统计信息
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())  # 缺失值填充

# 性别转换
data_test.ix[data_test['Sex'] == 'male', 'Sex'] = 0
data_test.ix[data_test['Sex'] == 'female', 'Sex'] = 1

# 上传地点转换
# data1 = data.groupby('Embarked').size() 由于数据中S占大多数所以将NaN赋值为S
data_test['Embarked'] = data_test['Embarked'].fillna('S')
data_test.ix[data_test['Embarked'] == 'S', 'Embarked'] = 0
data_test.ix[data_test['Embarked'] == 'C', 'Embarked'] = 1
data_test.ix[data_test['Embarked'] == 'Q', 'Embarked'] = 2

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# 决策树模型
alg = RandomForestClassifier(n_estimators=100,
                             min_samples_split=4,
                             min_samples_leaf=2,
                             random_state=1)  # 可以通过改变参数来提高模型的效果

KF = KFold(n_splits=3, random_state=1) # 将数据拆分成三等分
KF.get_n_splits(data_train[predictors].values)

predict_score = cross_val_score(alg,  # 用于拟合数据的对象
                                data_train[predictors].values,  # X
                                data_train['Survived'].values,  # Y
                                cv=KF)  # 交叉验证数

data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch']  # 组合新的特征
data_train['NameLength'] = data_train['Name'].apply(lambda x: len(x))  # 求每个元素的长度


def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    else:
        return ''


if __name__ == '__main__':

    titles = data_train['Name'].apply(get_title)
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6,
                     'Major': 7, 'Col': 7, 'Mlle': 8, 'Mme': 8, 'Don': 9, 'Lady': 10,
                     'Countess': 10, 'Jonkheer': 10, 'Sir': 9, 'Capt': 7, 'Ms': 2}

    for i in range(0, len(titles)):
        if title_mapping.get(titles[i]):
            titles[i] = title_mapping.get(titles[i])
        else:
            titles[i] = ''
    data_train['Title'] = titles  # 衍生出新的特征

    # matplotlib找出最具有代表性的特征以可视化界面形式展现
    predictorss = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                   'Fare', 'Embarked', 'FamilySize', 'Title',
                   'NameLength']

   # 选择K个最好的特征，返回选择特征后的数据
    select = SelectKBest(f_classif, k=5)  # 根据k个最高分选择功能，可以通过分值选择出最佳的特征

    select.fit(data_train[predictorss].values,
               data_train['Survived'].values)  # 拟合模型

    scores = -np.log10(select.pvalues_)  # 计算出每个特征的权重值，特征分数的p值

    plt.bar(range(len(predictorss)), scores)  # 柱状图

    plt.xticks(range(len(predictorss)), predictorss, rotation='vertical')

    # plt.show()

    # 通过图表展示其中权重比较大的几个特征：['Pclass', 'Sex', 'Fare', 'Title']
    # 两个不同的模型进行集成
    algorithms = [[GradientBoostingClassifier(n_estimators=50, random_state=1, max_depth=3),
                   ['Pclass', 'Sex', 'Fare', 'Title']
                   ],
                  [LogisticRegression(random_state=1),
                   ['Pclass', 'Sex', 'Fare', 'Title']
                   ]]

    KF1 = KFold(n_splits=5, random_state=1) # 将数据拆分成五等份

    predictions = []
    for train, test in KF1.split(data_train):  # 数据进行交叉验证
        train_target = data_train['Survived'].ix[train]  # 取出训练集的label
        full_test_predictions = []
        for alg, predictors in algorithms:  # 循环遍历不同的模型计算
            alg.fit(data_train[predictors].ix[train], train_target)  # 模型拟合
            test_predictions = alg.predict_proba(
                data_train[predictors].ix[test].astype(float))  # 预测模型 参数一预测为0的概率，参数二预测为1的概率
            full_test_predictions.append(test_predictions)  # 将多个不同模型计算出来的数值保存到list
        test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2  # 计算平均值
        test_predictions[test_predictions <= .5] = 0  # 评分小于0.5赋值0
        test_predictions[test_predictions > .5] = 1  # 评分大于0.5赋值1
        predictions.append(test_predictions)
    predictions = np.concatenate(predictions, axis=0)
    sum = 0
    for i in range(len(predictions)):
        if int(predictions[i][1]) == data_train['Survived'][i]:
            sum += 1
    accuracy = sum / len(predictions)  # 预测正确的值
    print(accuracy)
