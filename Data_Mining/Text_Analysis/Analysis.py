import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


class TF_IDF(object):

    def Data_Cleaning(self,cut_file_name,stop_file_name):
        '''
        对文件进行分词处理
        :param file_name: 分词文件名称
        :param stopfile_name: 停用词文件名称
        :return:
        '''
        data = pd.read_table(cut_file_name,names=['category','theme','URL','content'],encoding='utf-8')
        content = data['content'].values.tolist()
        content_S = []
        # 对每条记录集进行循环遍历分词
        for line in content:
            # 其中cut返回的是一个可迭代的对象,lcut返回的是一个列表
           semp  = jieba.lcut(line)
           # 去掉一些杂乱数据
           if len(semp) > 1 and semp != '\r\n':
              content_S.append(semp)
        # 将数据做成DataFrame类型
        df_content = pd.DataFrame({'content_S':content_S})
        # 建立停用词---对于那些出现频繁但是对文本分析没有任何用处的词语进行删除
        stop_word = pd.read_csv(stop_file_name,
                                index_col=False,
                                sep="\t",
                                quoting=3,
                                names=['stopword'],
                                encoding='utf-8')
        # 将数据转成字符串形式
        contents = df_content['content_S'].values.tolist()
        stopwords = stop_word['stopword'].values.tolist()
        contents_clean = self.drop_stopwords(contents, stopwords)
        # 建立新的词库
        df_train = pd.DataFrame({'contents_clean': contents_clean, 'label': data['category']})
        label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4,
                         "体育": 5, "教育": 6, "文化": 7, "军事": 8, "娱乐": 9, "时尚": 0}

        df_train['label'] = df_train['label'].map(label_mapping)

        x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values,
                                                            df_train['label'].values,
                                                            0.3,
                                                            random_state=1)
        # 对训练集数据进行格式变换
        words = []
        for line in range(len(x_train)):
            try:
                word = ' '.join(x_train[line])
                words.append(word)
            except:
                print(line, x_train['line'])

        return  words,x_train, y_train, x_test, y_test

    def drop_stopwords(self,contents, stopwords):
        '''
        基于停用词库清洗原先已分词好的df_content库，建立新的词库
        :param contents: 待处理分词后的词
        :param stopwords: 停用词
        :return:清洗后的文本
        '''
        contents_clean = []
        for line in contents:
            line_clean = []
            for word in line:
                # 如果word是停用词则跳出当前循环
                if word in stopwords:
                    continue
                # 将不是停用词的词语加入列表中
                line_clean.append(word)
            contents_clean.append(line_clean)
        return contents_clean


    def model_predict(self,words, x_test, y_train, y_test):
        '''
        基于模型进行训练
        :param words:进行预处理后的分词数据
        :param x_train: 训练标签数据集
        :param x_test: 测试标签数据集
        :param y_train: 训练标注数据集
        :param y_test: 测试标注数据
        :return:结果预测
        '''
        # 基于CountVectorizer贝叶斯模型进行模拟训练
        vec = CountVectorizer(analyzer='word',max_features=4000,lowercase = False)
        vec.fit(words)
        classifier = MultinomialNB()
        classifier.fit(vec.transform(words),y_train)

        # 对训练集数据进行格式变换
        test_words = []
        for line in range(len(x_test)):
            try:
                word = ' '.join(x_test[line])
                words.append(word)
            except:
                print(line,x_test['line'])

        vec_score = classifier.score(vec.fit(test_words),y_test)
        print(vec_score)

        # 基于TF-IDF贝叶斯进行模拟训练
        vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
        vectorizer.fit(words)
        classifier_1 = MultinomialNB()
        classifier_1.fit(vectorizer.transform(words), y_train)
        tf_idf_score = classifier_1.score(vectorizer.transform(test_words),y_test)
        print(tf_idf_score)


if __name__ == '__main__':

    cut_file_name = 'val.txt'
    stop_file_name = 'stopwords.txt'
    TI = TF_IDF()
    words, x_train, y_train, x_test, y_test = TI.Data_Cleaning(cut_file_name,stop_file_name)
    TI.model_predict(words,y_train, x_test, y_test)
