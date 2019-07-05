import re

import requests
from bs4 import BeautifulSoup
import pdfkit
import os
import logging
import warnings
warnings.filterwarnings('ignore')

class Land_and_land(object):

    def __init__(self):
        self.header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'
        }
        self.html_template = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                  <meta charset="UTF-8">
                </head>
                <body style = "width :50% ;margin:auto">
                {content}
                </body>
                </html>
                """
        self.page_num = '' # 总页数
        self.content2 = [] # 子页面链接
        self.content3 = []
        self.html_list = [] # 存储所有html文件名称
        self.pdf_list = [] # 存储所有pdf文件名称
        self.dir_pdf = 'D:/data_mining/ArticleSpider/ArticleSpider/spiders/人大数据清洗/pdf'
        self.dir_name = 'D:/data_mining/ArticleSpider/ArticleSpider/spiders/人大数据清洗/国土文件'
        self.config = 'D:/软件下载/wkhtmltox-0.12.5-1.mxe-cross-win64/wkhtmltox/bin/wkhtmltopdf.exe'

    def changeDir(self):
        """目录切换"""
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        os.chdir(self.dir_name)


    def get_one_page(self,url):
        '''获取第一页元素数据'''
        try:
            response = requests.get(url, headers=self.header)  # 该网站需要伪造一个身份才能访问
            if response.status_code == 200:
                response.encoding = 'utf-8'
                return response.text
            return None
        except :
            return None

    def page_one_nums(self,context):
        '''获取需要爬取的总页数'''
        soup = BeautifulSoup(context, 'lxml')
        # 获取总页数
        page_numbers = soup.select('#wp_page_numbers ul li')
        # 获取倒数第三位置的li标签
        numbers = page_numbers[len(page_numbers) - 3]
        num = numbers.get_text()  # 需要爬取的总页数
        return  num

    def page_one_page(self,context):
        '''解析页面元素获取该页面的子链接'''
        soup = BeautifulSoup(context,'lxml')
        page_context = soup.select('.ym-g66.ym-gl table tr td strong')
        for strong in page_context:
            strong_name = strong.get_text() # 公告名称
            href = 'http://www.sz68.com/' + strong.a.attrs['href'] # 公告链接
            dict = {
                'strong_name' :strong_name,
                'href': href
            }
            self.content2.append(dict)

    def page_two_page(self,strong_names,text):
        '''生成HTML文件'''
        soup = BeautifulSoup(text, 'html.parser')
        page_context = soup.select('.content')[0]  # 文本信息
        # 标题加入到正文的最前面，居中显示
        page_context = str(page_context)
        html = self.html_template.format(content=page_context)  # 将主题内容放到自定义模板中
        # 将文件中所有的字体样式替换成是宋体
        pattern = 'font-family:(.*?);'
        list = re.compile(pattern).findall(html)
        for res in list:
           if res == '宋体':
              pass
           elif res != '宋体':
               html = html.replace(res,'宋体')
        html = html.encode('utf-8') # 将bytes转成str类型
        try:
            with open(strong_names, 'wb') as f:
                f.write(html)
            return strong_names
        except Exception as e:
            logging.error('解析错误', exc_info=True)

    def save_pdf(self,html_url, pdf_url):
        """
        把所有html文件保存到pdf文件
        :param htmls:  html文件列表
        :param file_name: pdf文件名
        :return:
        """
        options = {
            'page-size': 'Letter',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': 'UTF-8',
            'custom-header': [
                ('Accept-Encoding', 'gzip')
            ],
            'cookie': [
                ('cookie-name1', 'cookie-value1'),
                ('cookie-name2', 'cookie-value2'),
            ],
            'outline-depth': 10,
        }

        # 传入一个html文件，生成一个PDF文件
        confg = pdfkit.configuration(wkhtmltopdf=self.config)
        html_file = self.dir_name + '/' + html_url # 文件的绝对路径
        pdfkit.from_file(html_file,
                         self.dir_pdf + '/' + pdf_url,
                         options=options,
                         configuration= confg)

    def main(self):
      self.changeDir()  # 判断是否存在该指定文件夹
      url = 'http://www.sz68.com/b/tdgg?s=0'  # 国土土地交易公告链接
      content = self.get_one_page(url)
      num = self.page_one_nums(content) # 获取总页数

      for i in range(0,int(num)-85):
          content1 = self.get_one_page('http://www.sz68.com/b/tdgg?s=%s'% i)
          self.page_one_page(content1) # 爬取子链接

      for contents in self.content2:
          strong_names = contents.get('strong_name') + '.html' # 子页面title
          hrefs = contents.get('href')  # 子页面链接
          text = self.get_one_page(hrefs)  # 子页面文本信息
          self.page_two_page(strong_names,text)
          self.html_list.append(strong_names)

      for name in self.html_list:
          self.save_pdf(name , name.split('.html')[0] + '.pdf')
          self.pdf_list.append(name.split('.html')[0] + '.pdf')

if __name__== '__main__':

    land = Land_and_land()
    land.main()
