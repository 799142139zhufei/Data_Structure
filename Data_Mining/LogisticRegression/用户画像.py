import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.ensemble import RandomForestClassifier  # 随机深林
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# 业务场景描述：通过用户的行为数据预测用户的年龄、学历和性别

tarin_querylist = 'D:\百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)' \
                  '/29-数据挖掘案例/用户画像/用户画像/data/train_querylist_writefile-1w.csv'

test_querylist = 'D:\百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)' \
                 '/29-数据挖掘案例/用户画像/用户画像/data/test_querylist_writefile-1w.csv'


# 由于gensim中的word2vec数据形式是list-of-list故加工成双层list形式并且建立model
def data_querylist(filename, data_file):
    with open(filename, 'r') as f:
        lines = f.readlines()  # 读取所有的行
        my_list = []
        for line in lines:
            cur_list = []
            line = line.strip()
            data = line.split(' ')
            for d in data:
                cur_list.append(d)
            my_list.append(cur_list)
        model = word2vec.Word2Vec(my_list, size=300, window=10, workers=4)  # 指定特征维度向量是size=300
        savepath = data_file + '_1w_word2vec' + '300' + '.model'
        model.save(savepath)
        # print(model.most_similar('中国人民大学')) # 能够匹配出相似的

# 加载训练好的word2vec模型，求用户搜索结果的平均向量
def cur_model(file_name, model_filename):
    cur_model = word2vec.Word2Vec.load(model_filename)
    with open(file_name, 'r') as f:
        cur_index = 0
        lines = f.readlines()
        doc_cev = np.zeros((len(lines), 300))
        for line in lines:
            word_vec = np.zeros((1, 300))
            words = line.strip().split(' ')
            wrod_num = 0
            # 求模型的平均向量
            for word in words:  # 每一行中的单个词进行遍历
                if word in cur_model:
                    wrod_num += 1
                    word_vec += np.array([cur_model[word]])  # 获取特征向量集
            doc_cev[cur_index] = word_vec / float(wrod_num)  # 对每个向量中数据集求平均值，但是最终形成的向量机是list-of-list
        cur_index += 1
        return doc_cev


train_cev = cur_model(tarin_querylist, 'train_1w_word2vec300.model')

train_cev = pd.DataFrame(train_cev)  # 将numpy转换成DataFrame

# 获取label标签

train_age = pd.read_csv('D:\百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)' \
                        '/29-数据挖掘案例/用户画像/用户画像/data/train_age.csv', names=['age_label'])

# train_education = pd.read_csv('D:\百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/
# 29-数据挖掘案例/用户画像/用户画像/data/train_education.csv')
# train_gender = pd.read_csv('D:\百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)
# /29-数据挖掘案例/用户画像/用户画像/data/train_gender.csv')


# 由于label标签中的0是一些未知的分类故将其删除掉
train = pd.concat([train_age, train_cev], axis=1)
train = train[train.ix[:, 'age_label'] != 0]
train_age = train['age_label']
age_train = train.drop('age_label', axis=1)


# EDA（数据探索性可视化分析）
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
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 测试集的构造方法和训练集一样做同样数据预处理

# data_querylist(test_querylist,'test')
# test_cev = cur_model(test_querylist,'test_1w_word2vec300.model')

# 建立预测模型

random_state = 7
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(age_train,
                                                    train_age,
                                                    test_size=test_size,
                                                    random_state=random_state)

# 构造逻辑回归模型
model = LogisticRegression()
model.fit(x_train.values, y_train.values)
test_predict = model.predict(x_test)
# print(model.score(x_test,y_test))
cnf_matrix = confusion_matrix(y_test, test_predict)

print('Recall-metric:',
      cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))  # 召回率（真实值时真-预测值时假）

print('accuracy-metric:',
      (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                  cnf_matrix[0, 1] + cnf_matrix[1, 1] + cnf_matrix[1, 0] + cnf_matrix[0, 0]))  # 准确率

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Gender-Confusion matrix')
plt.show()

# 构造随机深林模型
RF_model = RandomForestClassifier(n_estimators=100,  # 100个树
                                  min_samples_split=5,  # 拆分的最小深度
                                  max_depth=10)  # 最大的树节点
RF_model.fit(x_train, y_train)
test_predict = RF_model.predict(x_test)
print(RF_model.score(x_test, y_test))
cnf_matrix = confusion_matrix(y_test, test_predict)

print('Recall-metric:',
      cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))  # 召回率

print('accuracy-metric:',
      (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                  cnf_matrix[0, 1] + cnf_matrix[1, 1] + cnf_matrix[1, 0] + cnf_matrix[0, 0]))  # 准确率

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Gender-Confusion matrix')
plt.show()
