from gensim import corpora,models,similarities
import jieba
from gensim.models import word2vec
import warnings
warnings.filterwarnings('ignore')

word2vec.Word2Vec()
class Text_Mining(object):
      '''
      适用业务场景：存在多个文档时找出某一个文档与其他几个文档那个最匹配相识度最高，这是无监督学习的模型算法。
      '''
      def __init__(self):
          '''
          :param stopwordpath: 停用词路径（自己维护）
          :param file_text: 所有文件词频
          :param new_text: 输入需要预测的文档
          '''
          self.stopwordpath = '停用词.txt'
          self.ys_list = ['工作报告1.txt', '工作报告2.txt','工作报告3.txt','工作报告4.txt']
          self.new_list = ['整改报告.txt']
          self.file_text = []
          self.new_text = []
          self.docments = []
          self.all_list = []

      def stopwordslist(self):
          '''维护停用词用于去掉file_text和new_text中包含的停用词，可以有效的提高文档相似度
          :return: list[]
          '''
          stopwords = [line.strip() for line in open(self.stopwordpath, 'r', encoding='utf-8').readlines()]
          return stopwords

      def cut_text(self,filepath):
          '''文本分词
          :param filepath: 文件路径
          :return: list[]
          '''
          #1、读取文档信息
          doc = open(filepath, 'r', encoding='utf-8').read()
          texts = ''
          #2、对文档信息进行分词并处理成指定格式进行计算
          for text in jieba.cut(doc):
              texts += text + ' '
          return texts

      def del_stopword(self,docments):
          '''去掉文件中的停用词
          :return: list[]
          '''
          # 3、去掉停用词
          for docment in docments:
              text1 = []
              for word in str(docment).split():
                  if word not in self.stopwordslist():
                      text1.append(word)
              self.file_text.append(text1)

      def tf_idf(self,name,new_list):
          '''建立语料库、以及最终得到相似度结果
          :return: list[]
          '''
          # 4、通过语料库建立词典
          dictionary = corpora.Dictionary(self.file_text)
          # 5、加载要对比的文档
          new_doc = self.cut_text(new_list)
          #6、将要对比的文档通过doc2bow转化为稀疏向量
          new_vec = dictionary.doc2bow(new_doc.split())
          #7、对稀疏向量进行进一步处理，得到新语料库
          corpus = [dictionary.doc2bow(texts) for texts in self.file_text]
          #8、新语料库通过TfidfModel进行处理，得到tfidf
          tfidf = models.TfidfModel(corpus)
          # 9、通过token2id得到特征数 一共有多少个词、获取所有的文档中的词语（去重后）
          featurenum = len(dictionary.token2id.keys())
          # 10、稀疏矩阵相似度，从而建立索引
          index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=featurenum)
          # 11、得到最终相似度结果
          sim = index[tfidf[new_vec]]
          sim_list = sorted(enumerate(sim), key=lambda item: -item[1]) # 按最大相似度进行降序排列
          index_num = sim_list[0][0]
          ys_name = self.ys_list[index_num]
          num = sim_list[0][1]
          dict = {
              'list_sim': sim_list,  # 整改报告与子所有工作报告的相似度值
              'ys_name': ys_name,  # 整改报告与之最相识的工作报告名称
              'num': num,  # 最大的相识度值，也就是与整改报告最匹配的工作报告
              'name': name  # 对比的整改报告名称
          }
          self.all_list.append(dict)


      def main(self):
           for new_file in self.new_list:
              for file_path in self.ys_list:
                  docment = self.cut_text(file_path)
                  self.docments.append(docment)
              self.del_stopword(self.docments)
              name = new_file.split('.')[0]
              self.tf_idf(name,new_file)
           print(self.all_list)


if __name__ == '__main__':

   TM =  Text_Mining()
   TM.main()

