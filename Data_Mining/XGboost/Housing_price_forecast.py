import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm,skew
from scipy import stats
from scipy.special import boxcox1p
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import  GradientBoostingRegressor
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder, RobustScaler
import datetime
import warnings
warnings.filterwarnings('ignore')


'''
一、分析数据指标
    不同指标对结果的影响
    连续值和离散值的情况
二、观察数据正太性
    是否满足正太分布
    数据变换操作
三、数据预处理
    缺失值处理
    特征变换
四、集成法模型对比
    单模型回归效果
    平均和堆叠效果对比
'''
class Define_Model(object):

    def __init__(self):
        '''
        涉及到的模型
        '''
        #1、Lasso线性回归正则化是基于L1范数的;可以实现特征稀疏，去掉一些没有信息的特征
        self.lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

        #2、ElasticNet是岭回归和Lasso回归的融合，利用了L1和L2范数
        self.ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

        #3、岭回归是在损失函数中加入L2范数惩罚项来控制模型的复杂度；防止模型为了迎合模型训练而过于复杂出现过拟合能力，提高模型的泛化能力。
        self.KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

        #4、渐变提升回归
        self.GBoost = GradientBoostingRegressor(n_estimators=3000,
                                           learning_rate=0.05,
                                           max_depth=4,
                                           max_features='sqrt',
                                           min_samples_leaf=15,
                                           min_samples_split=10,
                                           loss='huber',
                                           random_state =5)

        #5、梯度提升树
        self.model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, #构造每个树时列的子采样率
                                     gamma=0.0468, #在树的叶节点上进行进一步分区所需的最小损耗减少量
                                     learning_rate=0.05, # 提高学习率
                                     max_depth=3, # 基础学习者的最大树深度
                                     min_child_weight=1.7817, # 子项中所需的实例权重（粗体）的最小总和
                                     n_estimators=2200, # 要拟合的提升树数
                                     reg_alpha=0.4640, # 权重上的L1正则项
                                     reg_lambda=0.8571, #权重的L2正则项
                                     subsample=0.5213, # 训练实例的子采样率
                                     n_jobs = -1) # 用于运行xgboost的并行线程数


    def Data_Cleaning(self):
        '''
        对整体进行数据清洗
        :return: x_train、y_train训练集
        '''
        train = pd.read_csv('train.csv')  # 训练集
        test = pd.read_csv('test.csv')  # 测试集

        # EDA-数据可视化探索性分析
        '''
        Data_GrLivArea = train.ix[:,['GrLivArea','SalePrice']]
        Data_GrLivArea.plot.scatter(x= 'GrLivArea',y= 'SalePrice',ylim=(0,800000))
        plt.show()

        Data_TotalBsmtSF = train.ix[:,['TotalBsmtSF','SalePrice']]
        Data_TotalBsmtSF.plot.scatter(x='TotalBsmtSF',y='SalePrice',ylim=(0,800000))
        plt.show()

        Data_OverallQual = train.ix[:,['OverallQual','SalePrice']]
        Data_OverallQual.plot.scatter(x='OverallQual',y='SalePrice',ylim=(0,800000))
        plt.show()

        # 同时查看多个特征的数据分布情况
        sns.set()
        cols = ['SalePrice', 'OverallQual', 'GrLivArea', 
                'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        sns.pairplot(train[cols], size = 2.5)
        plt.show()

        '''

        # 查看缺失值情况
        total = train.isnull().sum().sort_values(ascending=False)  # 对各个列中的null值进行降序排列
        featrue_list = total[total > 0].index.tolist()  # 获取存在缺失值的列
        total = train[featrue_list].isnull().sum().sort_values(ascending=False)
        percent = (train[featrue_list].isnull().sum() /
                   train[featrue_list].isnull().count()).sort_values(ascending=False)  # 各个列中的null值数/记录总数
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # 缺失值的占比

        # 对于训练集、测试集中的数据ID是没用价值的
        train.drop('Id', axis=1, inplace=True)
        test.drop('Id', axis=1, inplace=True)

        # 删除那些离群点数据
        train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index, axis=0)

        '''
        由于特征列不满足正太分布将其进行数据变换（对数变换）数据收敛
        '''
        # 未进行对数变换前
        sns.distplot(train['SalePrice'], fit=norm)
        (mu, sigma) = norm.fit(train['SalePrice'])

        # 分布图----展示正太分布效果
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')
        # plt.show()

        # QQ图----曲线越是与直线靠近说明正太分布效果越好
        fig = plt.figure()
        res = stats.probplot(train['SalePrice'], plot=plt)
        # plt.show()

        # 对数变换后
        train['SalePrice'] = np.log1p(train['SalePrice'])  # 数据值整体幅度变小

        sns.distplot(train['SalePrice'], fit=norm)
        (mu, sigma) = norm.fit(train['SalePrice'])

        # 分布图----展示正太分布效果
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')
        # plt.show()

        # QQ图----曲线越是与直线靠近说明正太分布效果越好
        fig = plt.figure()
        res = stats.probplot(train['SalePrice'], plot=plt)
        # plt.show()

        # 对缺失值数据进行处理
        cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage']
        train.drop(cols, axis=1, inplace=True)

        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            train[col] = train[col].fillna('None')

        gy_index = train['GarageYrBlt'].dropna().index  # 提取不存在缺失值的行
        train = train.ix[gy_index, :]

        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            train[col] = train[col].fillna('None')

        mvt_index = train['MasVnrType'].dropna().index  # 提取不存在缺失值的行
        train = train.ix[mvt_index, :]

        mva_index = train['MasVnrArea'].dropna().index  # 提取不存在缺失值的行
        train = train.ix[mva_index, :]

        el_index = train['Electrical'].dropna().index  # 提取不存在缺失值的行
        train = train.ix[el_index, :]

        # 通过LabelEncoder对不连续的数字或者文本进行编号
        cols = ['BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'BsmtFinType1',
                'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond',
                'YrSold', 'MoSold']
        for col in cols:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])

        # 查看各个特征的偏度值也就是不是符合正太分布情况
        numeric_feats = train.dtypes[train.dtypes != 'object'].index
        # 求每个列的偏度
        skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skewed_feats})

        skewness = skewness[abs(skewness) > 0.75]
        skewed_features = skewness.index
        lam = 0.15  # 关键点在于如何找到一个合适的参数，一般情况下0.15为经验值
        for feat in skewed_features:
            train[feat] = boxcox1p(train[feat], lam)

        x_train = train.drop('SalePrice', axis=1)
        y_train = train['SalePrice']
        x_train = pd.get_dummies(x_train)  # 对所有的特征列进行one-hot编码

        return  x_train,y_train


# 堆叠模型-----对多个模型的结果求平均值
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):
        self.models = models
        self.models_ = []

    # 触发这个函数是在调用cross_val_score()时x_train、y_train和分别传入进来
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models] # 将模型进行克隆
        for model in self.models_:
            model.fit(X, y)

    def predict(self, X):
        # 将几个模型的预测值整合到一个对象中
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

# 堆叠模型-----分二个阶段第一个阶段的输出作为下一个阶段的输入
class StackingAveragedModels(BaseEstimator,RegressorMixin,TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models # 第一阶段
        self.meta_model = meta_model # 第二阶段
        self.n_folds = n_folds # 交叉验证
        self.base_models_ = []

    def fit(self, X, y):
        X = pd.DataFrame(X)
        Y = pd.DataFrame(y.values)
        self.base_models_ = [list() for x in self.base_models] # [[],[],[]]
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156) # 等比例拆分
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models))) # 不同模型对数据特征集预测的结果
        for index, model in enumerate(self.base_models):  # 遍历每个模型
            for train_index, holdout_index in kfold.split(X,Y):
                instance = clone(model)
                self.base_models_[index].append(instance)
                instance.fit(X.iloc[train_index,:].values, Y.iloc[train_index,:]) # 拟合模型
                y_pred = instance.predict(X.iloc[holdout_index].values)
                out_of_fold_predictions[holdout_index, index] = y_pred
        # 参数一：三个模型的预测值；参数二：对应实际的真实值，通过模型进行拟合
        self.meta_model.fit(out_of_fold_predictions,Y)

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model.predict(meta_features) # 基于第二阶段拟合出来的模型预测第一阶段的特征

class ComputerEngineer(object):
    # 计算结果
    def rmsle_cv(model,x_train,y_train):
        n_folds = 5
        kf = KFold(n_splits=n_folds, random_state=42, shuffle=True).get_n_splits(x_train.values)
        rmse = np.sqrt(-cross_val_score(model,
                                        x_train.values,
                                        y_train,
                                        scoring='neg_mean_squared_error',
                                        cv=kf))
        return rmse

if __name__ == '__main__':

   start = datetime.datetime.now() # 开始执行时间

   dm = Define_Model()
   X_train,Y_train = dm.Data_Cleaning()

   averaged_models = AveragingModels(models=(dm.ENet, dm.GBoost, dm.KRR, dm.model_xgb, dm.lasso))
   score = ComputerEngineer.rmsle_cv(averaged_models,X_train,Y_train)
   end1 = datetime.datetime.now() # 结束时间
   print('Averaged base models score: %s (%s),程序执行时长：%s秒'%(round(score.mean(),4),
                                                                     round(score.std(),4),
                                                                    (end1 - start).seconds))

   stacked_averaged_models = StackingAveragedModels(base_models=(dm.ENet,dm.GBoost,dm.model_xgb),meta_model=dm.lasso)
   score = ComputerEngineer.rmsle_cv(stacked_averaged_models,X_train,Y_train)
   end12 = datetime.datetime.now() # 结束时间
   print('Stacking Averaged models score:  %s (%s),程序执行时长：%s秒'%(round(score.mean(),4),
                                                                          round(score.std(),4),
                                                                          (end12 - start).seconds))

