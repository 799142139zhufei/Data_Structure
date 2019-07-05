#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pymysql as py
import pandas as pd
from  sklearn.decomposition import PCA

#数据规约：对属性规约、数值规约
conn = py.connect(host='localhost',
                  port=3306,
                  user='root',
                  passwd='123456',
                  db='bootdo',
                  use_unicode=True,
                  charset="utf8")

sql = 'select hits,comment from myhexun'
data9=pd.read_sql(sql,conn)
ch=data9[u"comment"]/data9["hits"]
data9[u"评点比"]=ch

#--主成分分析进行中--
pca1=PCA()
pca1.fit(data9)
#返回模型中各个特征量
Characteristic=pca1.components_
#各个成分中各自方差百分比，贡献率
rate=pca1.explained_variance_ratio_
#print(rate)


pca2 = PCA(2)
pca2.fit(data9)
reduction = pca2.transform(data9)#降维
print(reduction)
recovery = pca2.inverse_transform(reduction)#恢复
#print(recovery)