import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import time
import warnings
warnings.filterwarnings('ignore')


class RFC(object):

     def Data_Preprocessing(self,file_name):
         '''
         数据预处理
         :param file_name: 文件名称
         :return: 对数据中各标签处理后的数据集
         '''
         # 1、创建数据集（预测球员出手投球是否会进球:label == shot_made_flag)
         data = pd.read_csv(file_name)
         # 特征变换将分钟转换成秒
         data['remaining_time'] = data['minutes_remaining'] * 60 + data['seconds_remaining']
         data['season_year'] = data['season'].apply(lambda x: int(x.split('-')[0]))
         # 生成新的列年份
         data = pd.concat([data, pd.get_dummies(data['season_year'], prefix='season_year')], axis=1)
         data['season_month'] = data['season'].apply(lambda x: int(x.split('-')[1]))
         # 生成新的列月份
         data = pd.concat([data, pd.get_dummies(data['season_month'], prefix='season_month')], axis=1)
         # 删除之前的列
         data.drop(['season_year', 'season_month'], axis=1,inplace=True)
         # 根据横坐标、纵坐标计算距离
         data['dist'] = np.sqrt(data['loc_x'] ** 2 + data['loc_y'] ** 2)

         # 通过图形化界面显示2个特征维度是正相关或者负相关只需要保留其中一个维度即可
         plt.subplot(133)
         plt.scatter(data['dist'], data['shot_distance'], color='blue', alpha=0.02)
         plt.title('dist and shot_distance')
         plt.show()

         plt.subplot(131)
         self.scatter_plot_by_category(data,'shot_zone_area')
         plt.title('shot_zone_area')

         plt.subplot(132)
         self.scatter_plot_by_category(data,'shot_zone_basic')
         plt.title('shot_zone_basic')

         plt.subplot(133)
         self.scatter_plot_by_category(data,'shot_zone_range')
         plt.title('shot_zone_range')
         plt.show()

         delete_columns = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range',
                 'shot_zone_basic', 'matchup', 'lon', 'lat', 'seconds_remaining', 'season',
                 'minutes_remaining', 'shot_distance', 'loc_x', 'loc_y', 'game_event_id',
                 'game_id', 'game_date']
         # 删除对应的列
         data = data.drop(delete_columns, axis=1)

         # 对部分列进行one-hot编码
         categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period']

         for categorical in categorical_vars:
             data = pd.concat([data, pd.get_dummies(data[categorical], prefix=categorical)], axis = 1)
             data = data.drop(categorical, axis=1)  # 删除已经转换后的列

         # 模拟训练集和测试集数据
         train_kobe = data[data['shot_made_flag'].notnull()]  # 剔除掉label中的缺失值（NaN）
         x_train = train_kobe.drop('shot_made_flag',axis = 1) # 标签
         y_train = train_kobe['shot_made_flag'] # 标注

         return x_train,y_train

     def Data_Visualization(self,data):
        '''
         数据可视化
         :param data:数据集
         :return: 图表
         '''
        alpha = 0.02 # 点的透明程度
        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.scatter(data['lon'],data['lat'],
                    color = 'blue',alpha=alpha) # 基于经度和维度
        plt.title('lon and lat')
        plt.subplot(132)
        plt.scatter(data['loc_x'],data['loc_y'],
                    color = 'green',alpha = alpha) # 基于横坐标和纵坐标
        plt.title('loc_x and loc_y')
        plt.show()

     def scatter_plot_by_category(self,data,feat):
        '''
         数据可视化
         :param data: 数据集
         :param feat: 特征标签
         :return: 图形
         '''
        alpha = 0.01 # 设置透明度
        gs = data.groupby(feat) # 基于该列分组后返回一个groupby对象,有几个类别就有几行
        cs = cm.rainbow(np.linspace(0, 1, len(gs))) # 生成6行4列的数据
        for g, c in zip(gs, cs):
            plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)


     def optimum_estimators(self,x_train,y_train):
        '''
         找到最合适的树节点
         :param x_train:训练集标签
         :param y_train:训练集标注
         :return:返回最佳的树节点
         '''
        min_score = 100000
        n_estimators = [i*10 for i in range(1,11)] # 树的个数
        KF1 = KFold(n_splits=5,shuffle=True,random_state=1)
        scores_n = []
        best_n = 0
        for n in n_estimators:
            print("the number of trees : {0}".format(n))
            t1 = time.time() # 开始时间
            rfc_score = 0 # 分数
            RFC = RandomForestClassifier(n_estimators=n)
            for train_c,test_c in KF1.split(x_train):
                X = x_train.iloc[train_c]
                Y = y_train.iloc[train_c]
                X1 = x_train.iloc[test_c]
                RFC.fit(X.values,Y)
                test_predict = RFC.predict(X1.values)
                rfc_score += log_loss(y_train.iloc[test_c].values,test_predict) / 10
            scores_n.append(rfc_score)
            if rfc_score < min_score:
                min_score = rfc_score
                best_n = n
            t2 = time.time() # 结束时间
            print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2-t1))
        return  best_n,min_score

     def optimum_depth(self,best_n,x_train,y_train):
        '''
         找到最适合树的最大深度
         :param x_train:训练集标签
         :param y_train:训练集标注
         :return: 返回最佳的树深度
         '''
        min_score = 100000
        max_depth = [i for i in range(1,11)] # 树d的最大深度
        KF2 = KFold(n_splits=5,shuffle=True,random_state=1)
        scores_m = []
        best_m = 0
        for m in max_depth:
            print("the number of max_depth : {0}".format(m))
            t1 = time.time() # 开始时间
            rfc_score = 0 # 分数
            RFC = RandomForestClassifier(n_estimators=best_n,max_depth=m)
            for train_c,test_c in KF2.split(x_train):
                X = x_train.iloc[train_c]
                Y = y_train.iloc[train_c]
                X1 = x_train.iloc[test_c]
                RFC.fit(X.values,Y)
                test_predict = RFC.predict(X1.values)
                rfc_score += log_loss(y_train.iloc[test_c].values, test_predict) / 10
            scores_m.append(rfc_score)
            if rfc_score < min_score:
                min_score = rfc_score
                best_m = m
            t2 = time.time() # 结束时间
            print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2-t1))
        return best_m, min_score

     def Model_Yuc(self,x_train,y_train,best_n,best_m):
        '''
         找出最适合的树个数（best_n）,最适合的深度（best_n）再进行预测模型
         :param x_train: 训练标签集
         :param y_train: 训练标注
         :param best_n: 最适合的树
         :param best_m: 最适合的深度
         :return: 预测结果
         '''
        X_train,X_test,\
        Y_train,Y_test = train_test_split(x_train.values,y_train,
                                          test_size=0.33,random_state=1)
        RFC1 = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
        RFC1.fit(X_train,Y_train)
        Test_score = RFC1.score(X_test,Y_test)
        print(Test_score)

if __name__ == '__main__':

    rfc = RFC()
    # 数据预处理
    x_train, y_train = rfc.Data_Preprocessing('../Text/NBA_DATA.csv')
    # 找到最合适的数
    best_n, min_score1 = rfc.optimum_estimators(x_train, y_train)
    print(min_score1)
    # 找到最合适的深度
    best_m, min_score2 = rfc.optimum_depth(best_n,x_train, y_train)
    print(min_score2)
    # 模型训练
    rfc.Model_Yuc(x_train, y_train,best_n,best_m)