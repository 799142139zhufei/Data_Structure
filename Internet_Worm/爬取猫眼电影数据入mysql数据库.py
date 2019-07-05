#coding=utf-8

import  requests
import re
from multiprocessing import Pool
import json
from requests.exceptions import RequestException
import pymysql
import datetime

#获取浏览链接并返回网页代码,添加异常捕捉机制防止程序中途停止终端
def get_one_page(url):
  try:
    header = {
         'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
    }
    response = requests.get(url,headers= header) #该网站需要伪造一个身份才能访问headers
    if response.status_code == 200:
       response.encoding = 'utf-8'
       return response.text
    return None
  except:
      return None

def parse_one_page(html):
    pattern = re.compile('<dd>.*?board-index.*?>(\d+)</i>.*?title="(.*?)".*?data-src="(.*?)".*?name">'
                         '<a.*?>(.*?)</a></p>.*?star">(.*?)</p>.*?releasetime">(.*?)'
                         '</p>.*?integer">(.*?)</i>.*?fraction">(.*?)</i></p>.*?</dd>',re.S) #获取则表达式字符
    items = re.findall(pattern,html)  #一次性全部打印出所有的匹配内容
    for item in items:
        yield {
            'index': item[0],
            'image_name':item[1],
            'image': item[2],
            'title': item[3],
            'actor': item[4].strip(), #去除掉换行符
            'time':item[5].strip(),
            'score':item[6]+item[7]
        }

def write_into_mysql(content):
    #print(content) 打印出来的是一个字典
    #print(content['index'])
    index =  str(content['index'])
    image_name = str(content['image_name'])
    image = str(content['image'])
    title = str(content['title'])
    actor = str(content['actor'])
    time = str(content['time'])
    score = str(content['score'])
    #同时将image图片爬取出来输出到本地目录下 C:\image
    response = requests.get(image)
    with open ('D:/image/maoyan_image/%s.jpg'% image_name,'wb') as f:
        f.write(response.content)
        f.close()
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d')  # 现在
    #获取数据库链接
    db = pymysql.connect(host='localhost',
                         port=3306,
                         user='root',
                         passwd='123456',
                         db='npc_test',
                         use_unicode=True,
                         charset="utf8")
     #获取游标链接
    cursor = db.cursor()
    # SQL 插入语句
    sql = "insert into parse_one_page(indexs,image,title,actor,times,score) " \
          "value ('%s', '%s', '%s','%s', '%s','%s')" % \
          (index,image,title,actor,time,score)
    #print(sql)
    #执行数据库sql语句
    cursor.execute(sql)
    # 提交
    db.commit()
    #游标关闭
    cursor.close()

def main(offset):
    url = 'http://maoyan.com/board/4?offset='+ str(offset)
    html = get_one_page(url)
    for itme in parse_one_page(html):
        write_into_mysql(itme)

if __name__ == '__main__':
    #for i in range(10):
         #main(i*10)
    pool =  Pool(4) #开启多个线程池
    result = pool.map(main,[i*10 for i in range(10)]) #爬取的主函数、传入main函数的入参
    #pool.close()


