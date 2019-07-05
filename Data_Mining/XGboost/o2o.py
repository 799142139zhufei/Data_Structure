#!/usr/bin/python3
# -*- coding:utf-8 -*- # python 2.0编码声明，不要忘记！
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.feature_selection import *
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

'''
开发流程：数据清洗、数据理解与分析、特征工程（数据预处理和特征选择）、模型的建立、模型评估
'''

# 1、数据集完整性验证:

# 检查京东用户数据（User）和京东行为数据（Action）是否一致

'''
data_user = pd.read_csv('D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/JData_User.csv')

user_id = data_user.loc[:,'user_id'].to_frame()
data_Action_02 = pd.read_csv('D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/JData_Action_201602.csv')
data_length2 =  pd.merge(user_id,data_Action_02,on = ['user_id','user_id'])
print(len(data_length2) == len(data_Action_02))

data_Action_03 = pd.read_csv('D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/JData_Action_201603.csv')
data_length3 = pd.merge(user_id,data_Action_03,on= ['user_id','user_id'])
print(len(data_length3) == len(data_Action_03))

data_Action_04 = pd.read_csv('D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/JData_Action_201604.csv')
data_length4 = pd.merge(user_id,data_Action_04,on= ['user_id','user_id'])
print(len(data_length4) == len(data_Action_04))
'''

# 检查用户行为数据(Action)中是否存在重复数

'''
data_Action_02 = pd.read_csv('D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/JData_Action_201602.csv')
data_Action_02.drop_duplicates(inplace=True)  # inplace=True删除所有重复项数据，inplace=False保留一条重复记录
'''

# 对用户数据（User）进行转换

'''
data_user = pd.read_csv('D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/JData_User.csv')
age_dict  = {
    'age':{
        '15岁以下':1,
        '16-25岁':2,
        '26-35岁':3,
        '36-45岁':4,
        '46-55岁':5,
        '56岁以上':6,
    }
}
data_user['age'] = data_user['age'].replace(age_dict)
'''

# 2、数据的清洗

'''
data_user = pd.read_csv('D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/User_table.csv')

# 删除那些age、sex

index = data_user[data_user['age'].isnull()].index
data_user = data_user['age'].drop(index,axis=0,inplace=True)

#删除无交互记录的用户

index_naction = data_user[(data_user['browse_num'].isnull()) 
                       & (data_user['addcart_num'].isnull()) 
                       & (data_user['delcart_num'].isnull()) 
                       & (data_user['buy_num'].isnull()) 
                       & (data_user['favor_num'].isnull()) 
                       & (data_user['click_num'].isnull())].index
                       
data_user = data_user.drop(index_naction,axis=0,inplace=True)

# 统计并删除无购买记录的用户

buy_index = data_user[data_user['buy_num'] == 0].index
data_user = data_user.drop(buy_index,axis = 0 ,inplace=True)

# 删除爬虫及惰性用户

bindex = data_user[data_user['buy_browse_ratio']<0.0005].index
data_user = data_user.drop(bindex,axis= 0,inplace=True)
'''

# 3、数据可视化探索分析（EDA）

'''
# 提取购买(type=4)的行为数据

def get_from_action_data(fname,chunk_size = 50000):
    reader = pd.read_csv(fname,iterator=True)
    loop = True
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[ ['user_id', 'sku_id', 'type', 'time'] ]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print('Iteration is stopped')
    df_ac = pd.concat(chunks,ignore_index=True)
    df_ac = df_ac[df_ac['type'] == 4]
    df_ac = df_ac[['user_id', 'sku_id', 'time']]
    return df_ac

df_ac = []
df_ac.append(get_from_action_data(fname = 'D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/JData_Action_201603_dedup.csv'))
df_ac.append(get_from_action_data(fname = 'D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/data/JData_Action_201604_dedup.csv'))

df_ac = pd.concat(df_ac,ignore_index=True)

# 将time字段转换为datetime类型
df_ac['time'] = pd.to_datetime(df_ac['time'])

# 使用lambda匿名函数将时间time转换为星期(周一为1, 周日为７)
df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)

# 周一到周日每天购买用户个数
df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['weekday', 'user_num']

# 周一到周日每天购买记录个数
df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['weekday', 'user_item_num']

# 周一到周日每天购买商品个数
df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['weekday', 'item_num']
'''

# 4、特征工程

'''
一、用户基本特征：
    获取基本的用户特征，基于用户本身属性多为类别特征的特点，对age,sex,usr_lv_cd进行独热编码操作，
    对于用户注册时间暂时不处理

二、商品基本特征：
    根据商品文件获取基本的特征
    针对属性a1,a2,a3进行独热编码
    商品类别和品牌直接作为特征

三、评论特征：
    分时间段，
    对评论数进行独热编码

四、行为特征：
    分时间段
    对行为类别进行独热编码
    分别按照用户-类别行为分组和用户-类别-商品行为分组统计，然后计算
    用户对同类别下其他商品的行为计数
    不同时间累积的行为计数（3,5,7,10,15,21,30）

五、累积用户特征：
    分时间段
    用户不同行为的
    购买转化率
    均值

六、用户近期行为特征：
    在上面针对用户进行累积特征提取的基础上，分别提取用户近一个月、近三天的特征，然后提取一个月内用户除去最近三天的行为占据一个月的行为的比重

七、用户对同类别下各种商品的行为:
    用户对各个类别的各项行为操作统计
    用户对各个类别操作行为统计占对所有类别操作行为统计的比重

八、累积商品特征:
    分时间段
    针对商品的不同行为的
    购买转化率
    均值

九、类别特征
    分时间段下各个商品类别的
    购买转化率
    均值
'''

# 5、建模

data = pd.read_csv('D:/百度网盘下载/20181211/14-人工智能阶段：-机器学习-深度学习-实战项目(1)/'
                        '29-数据挖掘案例/京东购买意向预测/train_set.csv')

data_x = data.loc[:,data.columns != 'label'] # 获取特征列
data_y = data.loc[:'label'] # 获取标签列

# 将数据集拆分为测试集、训练集
test_size = 0.33
random_state = 1
x_train,x_test,\
y_train,y_test = train_test_split(data_x,data_y,
                                  test_size=test_size,
                                  random_state=random_state)

x_val = x_test.iloc[:1500,:]
y_val = y_test.iloc[:1500,:]

x_test = x_test.iloc[1500:,:]
y_test = y_test.iloc[1500:,:]

x_train = x_train.drop(['user_id','sku_id'])
x_val = x_val.drop(['user_id','sku_id'])

dtrain = xgb.DMatrix(x_train,label=y_train)
dval = xgb.DMatrix(x_val,label=y_val)

param = {
         'n_estimators': 4000,
         'max_depth': 3,
         'min_child_weight': 5,
         'gamma': 0,
         'subsample': 1.0,
         'colsample_bytree': 0.8,
         'scale_pos_weight':10,
         'eta': 0.1,
         'silent': 1,
         'objective': 'binary:logistic',
         'eval_metric':'auc'
         }

num_round = param['n_estimators']

plst = param.items()
evallist = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=10)
bst.save_model('bst.model')