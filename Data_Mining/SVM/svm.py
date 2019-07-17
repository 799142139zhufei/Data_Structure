import numpy as np
from scipy import io as spio
from matplotlib import pyplot as plt
from sklearn import svm
import warnings
warnings.filterwarnings('ignore')

'''
支持向量机算法（该算法和逻辑回归类似是经典的二分类算法同时对多分类算法也适用）；
真正发挥作用的是数据点,也就是临界点；
核变换：在低纬度的情况下很难划分数据，可以通过将低维度的形式扩展到高维度，更好的对数据进行分类；
'''

class SVM(object):

    def model(self):
        '''data1——线性分类'''
        data1 = spio.loadmat('data1.mat')
        X = data1['X']
        y = data1['y']
        y = np.ravel(y)
        # 展示分布情况
        self.plot_data(X, y)

        model = svm.SVC(C=1.0, kernel='linear').fit(X, y)  # 指定核函数为线性核函数
        self.plot_decisionBoundary(X, y, model)  # 画决策边界

        '''data2——非线性分类'''
        data2 = spio.loadmat('data2.mat')
        X = data2['X']
        y = data2['y']
        y = np.ravel(y)
        plt = self.plot_data(X, y)
        plt.show()

        model = svm.SVC(gamma=100).fit(X, y)  # gamma为核函数的系数，值越大拟合的越好
        self.plot_decisionBoundary(X, y, model, class_='notLinear')  # 画决策边界

    # 作图
    def plot_data(self,X, y):
        '''
        未进行训练的模型
        :param X: 训练标签值
        :param y: 训练标注
        :return: 图形显示
        '''
        plt.figure(figsize=(10, 8))
        pos = np.where(y == 1)  # 找到y=1的位置
        neg = np.where(y == 0)  # 找到y=0的位置
        p1, = plt.plot(np.ravel(X[pos, 0]), np.ravel(X[pos, 1]), 'ro', markersize=8)
        p2, = plt.plot(np.ravel(X[neg, 0]), np.ravel(X[neg, 1]), 'g^', markersize=8)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend([p1, p2], ["y==1", "y==0"])
        return plt

    # 画决策边界
    def plot_decisionBoundary(self,X, y, model, class_='linear'):
        '''
        进行模型训练后的数据分布情况
        :param X: 训练标签值
        :param y: 训练标注
        :param model: 模型
        :param class_:
        :return: 图形显示
        '''
        plt = self.plot_data(X, y)
        # 线性边界
        if class_ == 'linear':
            w = model.coef_
            b = model.intercept_
            xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
            yp = -(w[0, 0] * xp + b) / w[0, 1]
            plt.plot(xp, yp, 'b-', linewidth=2.0)
            plt.show()
        else:  # 非线性边界
            x_1 = np.transpose(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100).reshape(1, -1))
            x_2 = np.transpose(np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).reshape(1, -1))
            X1, X2 = np.meshgrid(x_1, x_2)
            vals = np.zeros(X1.shape)
            for i in range(X1.shape[1]):
                this_X = np.hstack((X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1)))
                vals[:, i] = model.predict(this_X)
            plt.contour(X1, X2, vals, [0, 1], color='blue')
            plt.show()


if __name__ == "__main__":

    svm1 = SVM()
    svm1.model()
