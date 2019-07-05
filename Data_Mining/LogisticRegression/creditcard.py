#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import *

'''
业务场景描述：通过逻辑回归模型来预测信用卡交易数据异常检测
'''

# 数据可视化展示
'''
count_class = pd.value_counts(data['Class'],sort=True).sort_index() # 统计标签列class中的类别数量
count_class.plot(kind = 'bar')
plt.title('count_class')
plt.xlabel('xclass')
plt.ylabel('yclass')
plt.show()
'''

def plot_confusion_matrix(cm,
                          classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#Recall = TP/(TP+FN) 召回率
# 向下采样
def printing_Kfold_scores(X_train,Y_train):

    # 交叉验证将数据分成五份,迭代训练、测试
    fold = KFold(n_splits = 5,shuffle=False)  # 元素的总数、折叠数量、是否在分割成批次之前对数据进行洗牌
    c_param_range = [0.01,0.1,1,10,100] # 正则化惩罚值（用于逻辑回归模型训练）

    results_table = pd.DataFrame(columns=['C_parameter','score'])
    results_table['C_parameter'] = c_param_range # 展示惩罚值和评估分数

    j = 0
    # 循环遍历正则化惩罚值
    for c_param in c_param_range:

        print('------------开始--------------')
        print('C_param：' + str(c_param))
        print('------------------------------')
        print(' ')

        recall_accs = []
        # 其实将数据拆分成五份，每一份分为2块一块作为训练集一份作为测试集
        for iteration,indices in enumerate(fold.split(X_train),start=1):
            lr = LogisticRegression(C= c_param,penalty='l1')
            lr.fit(X_train.iloc[indices[0], :].values,
                   Y_train.iloc[indices[0], :].values.ravel()) # 拟合模型 ravel将多维数组降位一维

            y_pred_undersample = lr.predict(X_train.iloc[indices[1], :].values) # 预测模型

            recall_acc = recall_score(Y_train.iloc[indices[1], :].values,
                                      y_pred_undersample) # 评估分数
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ': recall score = ',recall_acc)

        results_table.ix[j,'score'] = np.mean(recall_accs) # 求出五份训练数据的的均值
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('------------结束-----------------')
        print('')

    if len(results_table['score'] == results_table['score'].max()) >1:
        best_c= results_table[0:1]['C_parameter']
    else:
        best_c = results_table[results_table['score'] == results_table['score'].max()]['C_parameter']
    print('C_parameter：'+ str(best_c))
    return best_c

# 过采样
def smote(data):

    columns = data.columns  # 获取数据的列
    features_columns = columns.delete(len(columns) - 1)  # 剔除掉最后一列class
    # data.drop('',axis=1) 基于列删除
    features = data[features_columns] # 特征列
    label = data['Class'] # 标签列

    features_train, features_test, \
    labels_train, labels_test = train_test_split(features,
                                                 label,
                                                 test_size=0.2,
                                                 random_state=0) # 训练数据测试数据比例拆分
    oversampler = SMOTE(random_state=0)

    os_features, os_labels = oversampler.fit_sample(features_train, labels_train) # 将训练数据扩增,测试数据保持原样

    os_features = pd.DataFrame(os_features) # 将数据类型转化成DataFrae类型
    os_labels = pd.DataFrame(os_labels) # 将数据类型转化成DataFrae类型

    best_c = printing_Kfold_scores(os_features, os_labels) # 利用训练集数据进行模型训练

    lr = LogisticRegression(C= float(best_c), penalty='l1')

    lr.fit(os_features.values, os_labels.values.ravel()) # 拟合

    y_pred = lr.predict(features_test.values) # 预测

    cnf_matrix = confusion_matrix(labels_test, y_pred) # 对比预测和真实情况（参数一：真实值；参数二：预测值）
    
    print('Recall metric in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])) # 召回率 = tp/tp+fp
    print('Accuracy metric in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1])) # 精确率 = tp/tp+fn

    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes=class_names,
                          title='Confusion matrix')
    plt.show()


if __name__ == '__main__':

    data = pd.read_csv('creditcard.csv')

    # 样本不均时，怎么处理:向下采样（class中0和1分布不均，使得正、负样本数据均衡）
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))  # 数据的标准化[-1,1]数据满足正太分布发散性更强
    data = data.drop(['Time', 'Amount'], axis=1)  # 删除无相关的列
    # print(data.head(5))

    x = data.ix[:, data.columns != 'Class']  # 获取特征
    y = data.ix[:, data.columns == 'Class']  # 获取标签

    number_records_fraud = len(data[data['Class'] == 1])  # class=1 获取个数
    fraud_indices = np.array(data[data['Class'] == 1].index)  # class=1 获取索引值
    normal_indices = data[data['Class'] == 0].index  # class = 0 获取索引值

    '''
    np.random.choice(a, size=None, replace=True, p=None)
    返回：从一维array a 或 int 数字a 中，以概率p随机选取大小为size的数据，
    replace表示是否重用元素，即抽取出来的数据是否放回原数组中，
    默认为true（抽取出来的数据有重复）
    '''
    random_normal_indices = np.random.choice(normal_indices,
                                             number_records_fraud,
                                             replace=False)  # 从class=0的数据中筛选出与class=1一致的个数

    random_normal_indices = np.array(random_normal_indices) # 生成class=0array数组类型索引值

    under_sample_indices = np.concatenate([fraud_indices,
                                           random_normal_indices])  # 获取class=1和class=0的index

    under_sample_data = data.iloc[under_sample_indices, :] # 数据清洗后（向下采样）的样本数据---数据样本减少

    X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class'] # 获取特征
    Y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class'] # 获取标签

    '''
       print('Percentage of normal transactions: ',
              len(under_sample_data[under_sample_data['Class'] == 1]) / len(under_sample_data))
        print('Percentage of fraud transactions: ',
              len(under_sample_data[under_sample_data['Class'] == 0]) / len(under_sample_data))
        print('Total number of transactions in resampled data: ', len(under_sample_data))
    
    '''
    # 交叉验证 （准确率和召回率需要整清楚）
    X_train, X_test, \
    Y_train, Y_test = \
        train_test_split(x,y,
                         test_size=0.3,
                         random_state=0)  # 对原始数据进行拆分，后期用于模型训练、测试验证

    X_train_undersample, X_test_undersample, \
    Y_train_undersample, Y_test_undersample = \
        train_test_split(X_undersample,
                         Y_undersample,
                         test_size=0.3,
                         random_state=0)  # 对数据处理后的向下取样数据进行拆分，后期用于模型训练

    # 向下采样
    best_c = printing_Kfold_scores(X_train_undersample,Y_train_undersample)

    # 向上采样
    smote(data)






