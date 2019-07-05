#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

from gensim.models import word2vec
import warnings
warnings.filterwarnings('ignore')

data = ['The richest city in China is Shenzhen', 'the poorest citys is Jingzhou']
cut_data = [s.split() for s in data]  # 拆分数据集
model = word2vec.Word2Vec(cut_data,min_count=1)
socre = model.similarity('China','city') # 对比两个词语的相似度越大相似度越高
print(socre)

