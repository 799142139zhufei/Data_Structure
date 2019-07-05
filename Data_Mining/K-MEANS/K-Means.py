import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import DBSCAN

'''
K-Means聚类算法（无监督学习算法）没有label；
a)、K值如何确定？
    都会影响到后续的结果b)、质心的确认？初始点的不同对得出的结论影响比较大，需要多次尝试得出最佳的初始点。

适用于相对规则的数据集

DBSCAN算法的优势：
a)、不需要指定K的值,通过指定R半径的大小发展下限的形式自己去找对应的簇；
b)、擅长找离群点（检测任务）；
c)、基本2个参数就OK了；

eps ---- 对应点的半径决定了每个簇的个数
min_samples --- 限定每个簇最少为多少个数

劣势：
a)、对于高维度的就难办了（PCA降维）；
b)、效率慢；
适用于相对不规则的数据集
'''

# 对啤酒数据进行无标签分类

data = pd.read_csv('data.txt', sep=' ')
X = data[list(data.ix[:, 1:].columns)]  # 获取指定的列
km1 = KMeans(n_clusters=3).fit(X)  # 将数据分为3簇
km2 = KMeans(n_clusters=2).fit(X)  # 将数据分为3簇
data['cluster1'] = km1.labels_  # 每个样本点的标签
data['cluster2'] = km2.labels_  # 每个样本点的标签

# print(km1.cluster_centers_)  聚类中心点的坐标
# print(data.groupby('cluster1').mean())  聚类中心点的坐标
# print(data.groupby("cluster2").mean())  聚类中心点的坐标
# centers = data.groupby("cluster1").mean().reset_index()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)  # 数据预处理---数值归一化
km = KMeans(n_clusters=3).fit(X_scaled)
data['scaled_cluster'] = km.labels_

score1 = metrics.silhouette_score(X, data['scaled_cluster'])  # 数值归一化后
score2 = metrics.silhouette_score(X, data['cluster1'])  # 数值为进行与处理

# 循环遍历找出最佳的K值
score = {}
for k in range(2, 10):
    labels = KMeans(n_clusters=k).fit(X).labels_
    scores = metrics.silhouette_score(X, labels)
    score[k] = round(scores, 5)

# DBSCAN算法模型
dbscan = DBSCAN(eps=10, min_samples=2).fit(X)
data['dbsan_label'] = dbscan.labels_
print(data.ix[:, 5:])
