#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import jieba
import jieba.posseg as ps
import jieba.analyse as al
'''
文本挖掘中的常用的分词操作；
jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的generator，
可以使用 for 循环来获得分词后得到的每一个词语(unicode)
for item in seg_list:
    print(type(item))
其中默认的是精准模式
'''

seg_list = jieba.cut('我在学习自然语言处理',cut_all=True)
#print('Full Mode：'+ '/'.join(seg_list)) # 全模式

def_list = jieba.cut('我在学习自然语言处理',cut_all=False)
#print('Default Mode: ' + '/'.join(def_list)) #精确模式

ss_list = jieba.cut_for_search('小明硕士毕业于中国科学院计算所，后在哈佛大学深造')
#print('Ss Mode: ' + '/'.join(ss_list)) #搜索引擎模式

#词性标注
'''
ps_list = ps.cut('我在学习自然语言处理')
for item in ps_list:
    print(item.word + '-----' + item.flag)
'''

#更改词频
'''
#add_word只能调高词频，不能调低词频，使用jieba.suggest_freq("词语",True)也一样
sen_list = '我喜欢上海东方明珠'
#jieba.add_word('我喜欢')
#jieba.suggest_freq('我喜欢',True)
sen_list1 = jieba.cut(sen_list)
for item in sen_list1:
    print(item)
    
'''

#提取关键词(提取词频较高排名前三的词频)
'''
sen_list2 = '我喜欢上海东方明珠'
tag = al.extract_tags(sen_list2,3)
print(tag)
'''

#返回词语的位置

'''
seg_list3 = '我喜欢上海东方明珠'
home_list = jieba.tokenize(seg_list3) #默认以精准模式的方式
for item in home_list:
    print(item)
print('')
home_list1 = jieba.tokenize(seg_list3,mode='search') #以搜索引擎的方式
for item in home_list1:
    print(item)
'''



#分析血尸词频：频率最高的前15位

data = open('D:/百度网盘下载/源码/源码/第7周/血尸1.txt','r',encoding='utf-8').read()
tag = al.extract_tags(data,15)
print(tag)


#盗墓笔记的关键词提取
'''
data1 = open('D:/百度网盘下载/源码/源码/第7周/盗墓笔记2.txt','r',encoding='utf-8').read()
tag1 = al.extract_tags(data1,30)
print(tag1)
'''
