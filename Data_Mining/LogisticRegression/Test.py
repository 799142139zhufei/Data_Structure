#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pandas as pd
import matplotlib.pyplot as plt

# 根据成绩情况是否被录取的人员分布图
data = pd.read_csv('LogiReg_data.csv',header = None,names=['achievement1','achievement2','lable'])

positive = data[data['lable'] ==1 ] # 等于1的为正
negative = data[data['lable'] ==0 ] # 等于0的为负

fig,ax = plt.subplots(figsize = (10,5))
ax.scatter(positive['achievement1'],positive['achievement2'],s=30,c='b',marker = 'o',label = 'lable')
ax.scatter(negative['achievement1'],negative['achievement2'],s=30,c='r',marker = 'x',label = 'not lable')
ax.legend() # 显示右上角的标签
ax.set_xlabel('achievement1 score') # 显示横坐标
ax.set_ylabel('achievement2 score') # 显示纵坐标
plt.show()