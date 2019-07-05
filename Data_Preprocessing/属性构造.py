#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pymysql as py
import pandas as pd

#属性构造：通过2个或者多个属性间存在相关性，构造出一个新的数据存入数据库中。
conn = py.connect(host='localhost',port=3306,user='root',passwd='123456',
                  db='bootdo',use_unicode=True,charset="utf8")

sql = 'select * from myhexun'
data = pd.read_sql(sql,conn)
#print(data)
ch = data[u'comment']/data['hits'] # 也可以算是一种特征衍生
#print(ch)
data[u'评点比'] = ch
#print(data)
file = './myhexun.xls'
data.to_excel(file,index=False)
