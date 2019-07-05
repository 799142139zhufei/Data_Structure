#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！


import pymysql as py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

'''
数据清洗步骤：
1、对于数据中的极大值、极小值情况进行处理；
2、特征的规格不一样，不能够放到一起进行比较，通过无量纲化进行处理。
'''

conn = py.connect(host='localhost',port=3306,user='root',passwd='123456',
                  db='bootdo',use_unicode=True,charset="utf8")
sql = 'select price,comment from taob'
data = pd.read_sql(sql,conn)

#1、对数据进行离差标准化--------消除了量纲及极大极小异常值的影响（区间缩放，返回值为缩放到[0, 1]区间的数据）
from sklearn.preprocessing import MinMaxScaler
data['comment1'] = MinMaxScaler().fit_transform(data['comment'].values.reshape(-1,1)).reshape(1,-1)[0]
#data2 = (data-data.min())/(data.max()-data.min())


#2、对数据进行标准差标准化(标准化，返回值为标准化后的数据[-1,1]满足正太分布)
from sklearn.preprocessing import StandardScaler
data['price'] = StandardScaler().fit_transform(data['price'].values.reshape(-1,1)).reshape(1,-1)[0]
#data3 = (data-data.mean())/data.std()

#3、Normalizer 归一化是依照特征矩阵的行处理数据；将整个特征行和列进行归一化处理对指定的列处理没意义
from sklearn.preprocessing import Normalizer
data = Normalizer(norm='l1').fit_transform(data)

#4、定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
from sklearn.preprocessing import Binarizer
data['price'] = Binarizer(threshold=100).fit_transform(data['price'].values.reshape(-1,1)).reshape(1,-1)[0]

#5、OneHotEncoder独热编码get_dummies()采用one-hot进行编码
from sklearn.preprocessing import OneHotEncoder
data['price'] = pd.get_dummies(data['price'])


#6、Imputer缺失值计算，返回值为计算缺失值后的数据,
#缺失值计算，返回值为计算缺失值后的数据
#参数missing_value为缺失值的表示形式，默认为NaN
#参数strategy为缺失值填充方式，默认为mean（均值）
from sklearn.preprocessing import Imputer
Imputer(missing_values=10,strategy='mean').fit_transform(data['price'])

#7、对数据进行小数定标规范化
k = np.ceil(np.log10(data.abs().max()))
print(k)
data4 = data/10**k
#print(data4)

#8、连续性数据离散化
#等宽离散化
data5 = data[u'price'].copy()
data6 = data5.values
k= 3 # 等宽区间；也可以指定宽度如：k = [1,100,1000,10000]分为三个区间1-100,100-1000,1000-10000
data7 = pd.cut(data6,k,labels=['很便宜','价格适中','有点贵'])
#print(data7)



