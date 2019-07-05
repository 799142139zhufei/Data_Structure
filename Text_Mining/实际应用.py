#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明

from gensim import corpora,models,similarities
import gensim
import jieba
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

'''
1、读取文档
2、对要计算的多篇文档进行分词
3、对文档进行整理成指定格式，方便后续进行计算
4、计算出词语的频率
5、对频率低的词语进行过滤（此操作可做可不做）
6、通过语料库建立词典
7、加载要对比的文档
8、将要对比的文档通过doc2bow转化为稀疏向量
9、对稀疏向量进行进一步处理，得到新语料库
10、将新语料库通过tfidfmodel进行处理，得到tf-idf
11、通过token2id得到特征数
12、稀疏矩阵相似度，从而建立索引
13、得到最终相似度结果
'''

#1、读取文档信息
doc1 = open('D:/百度网盘下载/源码/源码/第7周/工作报告1.txt','r',encoding='utf-8').read()
doc2 = open('D:/百度网盘下载/源码/源码/第7周/工作报告2.txt','r',encoding='utf-8').read()

#2、对文档进行分词
data1 = jieba.cut(doc1)
data2 = jieba.cut(doc2)

#3、对文档进行整理成指定格式，方便后续进行计算
data11 = ''
data12 = ''
for item in data1:
    data11+= item + ' '

for item in data2:
    data12 += item +' '

docments = [data11,data12]

# 去掉停用词
def stopwordslist():
    stopwords = [line.strip() for line in open('停用词.txt','r',encoding= 'UTF-8').readlines()]
    return stopwords

text = []
for docment in docments:
    text1 = []
    for word in docment.split():
        if word not in stopwordslist():
            text1.append(word)
    text.append(text1)


#4、计算出词语的频率 可以统计出每个词语出现的次数
frequeney = defaultdict(int)
for word in text:
    for words in word:
        frequeney[words] += 1

#5、对频率低的词语进行过滤(提取词语出现次数大于3次的)
wenben = []
for texts in text:
    for word in texts:
        if frequeney[word] > 3:
           wenben.append(word)

#6、通过语料库建立词典
dictionary = gensim.corpora.Dictionary(text) # 对多个list中的元素合并并去重
dictionary.save('D:/百度网盘下载/源码/源码/第7周/yuliaoku.txt')

#7、加载要对比的文档
doc3 = open('D:/百度网盘下载/源码/源码/第7周/整改报告.txt','r',encoding='utf-8').read()
data3 = jieba.cut(doc3)
data31 = ''
for item in data3:
 data31 += item + ' '
new_doc = data31

#8、将要对比的文档通过doc2bow转化为稀疏向量
new_vec = dictionary.doc2bow(new_doc.split())
#print(new_vec) # 列表中包含多个元组指的是每个词语出现的次数，第一个元素是key,第二个是出现的次数

#9、对稀疏向量进行进一步处理，得到新语料库
corpus = [dictionary.doc2bow(texts) for texts in text] # 列表中包含多个元组指的是每个词语出现的次数
#corpora = corpora.MalletCorpus.serialize('D:/百度网盘下载/源码/源码/第7周/mm.mm',corpus)

#10、将新语料库通过TfidfModel进行处理，得到tfidf
tfidf = models.TfidfModel(corpus)
#print(tfidf) 统计出2个word文档中出现相同的词语；比如文档A、B都出现了‘张三’,则 key='zhangsan';value=2

#11、通过token2id得到特征数 一共有多少个词
featurenum = len(dictionary.token2id.keys()) # 获取所有的文档中的词语（去重后）

#12、稀疏矩阵相似度，从而建立索引
index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featurenum)

#13、得到最终相似度结果
sim = index[tfidf[new_vec]]
print(list(enumerate(sim)))

