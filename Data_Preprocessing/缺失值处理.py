#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pymysql as ps
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
数据清洗步骤：
1、缺失值处理（通过describe与len直接发现，通过0数据发现）
2、异常值处理（通过散点图发现）
缺失值一般处理方式：删除、不处理、插补
'''

conn = ps.connect(host='localhost',port=3306,user='root',passwd='123456',
                  db='bootdo',use_unicode=True,charset="utf8")
sql = 'select * from taob'
data = pd.read_sql(sql,conn)

#print(data.describe())  #打印出对应的中位数、均值、平均值、标准差等

#缺失值处理
#print(data[data['price'] == 0]['title']) # 先筛选出满足price等于0的数据
x = 0
'''
for i in data.columns:
    for j in range(len(data)):
        if data[i].isnull()[j]:
            data[i][j] = 36
'''

'''
#异常值处理
#根据画散点图显示（横轴为价格，纵轴评论数）
price = data['price'].values
comment = data['comment'].values
plt.plot(price,comment,'*')
font_set = FontProperties(fname=r"C:\Windows\Fonts\simfang.ttf",size=12) #指定格式的中文字符输出
plt.title('趋势图',fontproperties=font_set)
plt.xlabel('价格',fontproperties=font_set)
plt.ylabel('评论数',fontproperties=font_set)
plt.show()

'''


# 均值填充极大值
line = len(data.values) # 行数
colums = len(data.columns) # 列数
datas = data.values
for i in range(0,line): # 遍历行
    for j in range(0,colums): # 遍历列
        if datas[i][2]>130:
            datas[i][2] = 36
        if datas[i][3] > 300:
            datas[i][3] = 58

dt = datas.T
price = dt[2]
comment = dt[3]
plt.plot(price,comment,'*')
plt.show()

#直方图分布情况
price_max = dt[2].max()
price_min = dt[2].min()

comment_max = dt[3].max()
comment_min = dt[3].min()


#极差：最大值-最小值
price_rg = price_max-price_min
comment_rg = comment_max-comment_min

#组距：极差/组数
price_st = price_rg/12
comment_st = comment_rg/12

#价格
price_ty = np.arange(price_min,price_max,price_st)
plt.hist(dt[2],price_ty)
plt.show()

#评论数
comment_ty = np.arange(comment_min,comment_max,comment_st)
plt.hist(dt[3],comment_ty)
plt.show()

