#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！


import numpy as np
import pandas as pd
import missingno as msno

'''
业务场景描述：针对足球赛事数据进行数据分析
'''

# 1、数据预处理
data = pd.read_csv('Text/redcard.csv')
all_columns = data.columns.tolist() # 获取所有的列
dtype_value = data.dtypes.value_counts() # 获取各个特征的分布情况
data_type = data.select_dtypes(include='object').columns.tolist() # 获取特征属性为object的列，对这些列需要进行特征变换


# 2、数据的切分模块

# 由于运动员中存在身高数据为NaN且多条重复的数据
data_height = data['height'].dropna(axis=0) # axis=0基于行
data = data.iloc[data_height.index,:]

height1 = data['height'].groupby(data['playerShort']).mean()
height1 = sum(height1)/len(height1)
height2 = data['height'].mean()
# print(height1,height2) 排除掉一些缺失值数据后均值存在差异，都会影响模型的好坏；建议使用height1

# 筛选出和运动员挂钩的特征
player_index = 'playerShort'
player_cols = ['birthday',
               'height',
               'weight',
               'position',
               'photoID',
               'rater1',
               'rater2',
              ]

# 基于运动员进行分组再对特征列进行去重确定是否是唯一值
all_cols_unique_players = data.groupby('playerShort').agg({col:'nunique' for col in player_cols}) # groupby基于行聚合，agg基于列聚合；
#print(all_cols_unique_players[all_cols_unique_players >1].dropna())

# 写成一个公用方法调用检查数据的完整性
def get_subgroup(data_play,data_index,data_cols):
    list = []
    g = data_play.groupby(data_index).agg({ col:'nunique' for col in data_cols })
    if g[g >1].dropna().shape[0] != 0:
        list.append(data_index)
    return  (data_play.groupby(data_index).agg({ col:'max' for col in data_cols }))

players = get_subgroup(data,player_index,player_cols)


# 球员和裁判的关系

dyad_index = ['refNum', 'playerShort']
dyad_cols = ['games',
             'victories',
             'ties',
             'defeats',
             'goals',
             'yellowCards',
             'yellowReds',
             'redCards',
            ]
dyads = get_subgroup(data,dyad_index,dyad_cols)

# 3、对于缺失值数据的处理

msno.matrix(players.sample(1000), labels=True)  # 无效数据密度显示
msno.bar(players.sample(1000))  # 条形图显示
msno.heatmap(players.sample(1000))  # 热图相关性显示
msno.dendrogram(players.sample(1000))  # 树状图显示

players['rater1'] = players[['rater1'].notnull()]
players['rater2'] = players[['rater2'].notnull()]

