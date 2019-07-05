from jieba import analyse # TextRank关键词抽取接口
import warnings
warnings.filterwarnings('ignore')

text = '鼓励社会资本投资或参股基础设施、公用事业和公共服务等领域项目'
textrank = analyse.textrank
keywords = textrank(text, # 文本
                    topK=5, # 排名前五
                    allowPOS=('ns', 'n', 'vn', 'a'), # 指定词性
                    withWeight=True) # list[]
for words,w in keywords:
    print(words + ',' + str(w))

