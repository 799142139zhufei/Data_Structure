#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_table('D:/百度网盘下载/20181015/课程资料/唐宇迪-机器学习课程资料/'
                   '机器学习算法配套案例实战/贝叶斯-新闻分类/贝叶斯-新闻分类/data/val.txt',
                   names=['category','theme','URL','content'],
                   encoding='utf-8')

data = data.dropna()
content = data['content'].values.tolist()

content_S = []
for line in content: # 对每条记录集进行循环遍历分词
   semp  = jieba.lcut(line) # 其中cut返回的是一个可迭代的对象,lcut返回的是一个列表
   if len(semp) > 1 and semp != '\r\n': # 换行符
      content_S.append(semp)

df_content = pd.DataFrame({'content_S':content_S})

# 建立停用词---对于那些出现频繁但是对文本分析没有任何用处的词语进行删除
stop_word = pd.read_csv('D:/百度网盘下载/20181015/课程资料/唐宇迪-机器学习课程资料/'
                        '机器学习算法配套案例实战/贝叶斯-新闻分类/贝叶斯-新闻分类/stopwords.txt',
                        index_col=False,
                        sep="\t",
                        quoting=3,
                        names=['stopword'],
                        encoding='utf-8')

# 基于停用词库清洗原先以分词好的df_content库，建立新的词库
def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords: # 如果word是停用词则跳出当前循环
                continue
            line_clean.append(word) # 将不是停用词的词语加入列表中
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words

contents = df_content['content_S'].values.tolist() # 将数据转成字符串形式
stopwords = stop_word['stopword'].values.tolist()  # 将数据转成字符串形式
contents_clean, all_words = drop_stopwords(contents, stopwords)

# 建立新的词库
df_content=pd.DataFrame({'contents_clean':contents_clean})
df_all_words=pd.DataFrame({'all_words':all_words})

df_train=pd.DataFrame({'contents_clean':contents_clean,'label':data['category']})

# lable = df_train['label'].unique() 对标签进行编码

label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4,
                 "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}

df_train['label']= df_train['label'].map(label_mapping)

x_train,y_train,x_test,y_test = train_test_split(df_train['contents_clean'].values,
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
        print(line,x_train['line'])

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