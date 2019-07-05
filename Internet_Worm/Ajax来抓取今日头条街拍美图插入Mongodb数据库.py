#coding=utf-8

import requests
from urllib.parse import urlencode
from requests.exceptions import RequestException
import json
from bs4 import BeautifulSoup
import pymongo
import re
import datetime

mongo_url = 'localhost'
mongo_port = 27017
mongo_db = 'toutiao'
mongo_table = 'toutiao'

client = pymongo.MongoClient(mongo_url,mongo_port)
db = client[mongo_db]

#获取文本信息
def get_page_index(offset,keyword):

        #json的请求参数 dict 元组
        data = {
                'offset': offset,
                'format': 'json',
                'keyword': keyword,
                'autoload': 'true',
                'count': '20',
                'cur_tab': 1,
                'from': 'search_tab',
                }

        #防止访问不通过伪造一个身份
        hearder = {
        'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
        }
        Request_URL =  'https://www.toutiao.com/search_content/?' + urlencode(data) #将data字典类型的数据转换成url的链接参数
        try:
                response = requests.get(Request_URL,headers = hearder)
                if response.status_code == 200:
                    response.encoding = 'utf-8'
                    return response.text
                return None
        except RequestException:
            print('请求失败')
            return None

#获取返回的json中data属性数据
def parse_page_index(html):
    data = json.loads(html) #将字符串转换成json对象
    if data and 'data' in data.keys():  #判断返回数据中包含data属性；data.keys()返回的是所有的键名
        for item in data.get('data'):
            yield item.get('article_url') #生成一个生成器

#遍历data
def get_page_detail(url):
    hearder = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
    }
    try:
        response = requests.get(url,headers = hearder)
        if response.status_code == 200:
            response.encoding = 'utf-8'
            return response.text
        return None
    except RequestException:
        print('请求图片链接失败',url)
        return None

#解析提取元素
def parse_page_detail(html,url):

    soup = BeautifulSoup(html,'lxml')
    title = soup.select('title')[0].get_text()
    restult = re.search('JSON.parse\("(.*?)"\),',html,re.S)
    if restult:
        data = json.loads(restult.group(1).replace('\\"','"'))  #将字符串转换成json对象
        if data and 'sub_images'in data.keys(): #判断data的所有key中是否包含sub_images
            sub_images = data.get('sub_images')
            images = [image.get('url').replace('\\/','/') for image in sub_images]
            spider_date = datetime.datetime.now() #数据抓取时间
            return {'url': url,
                    'title':title,
                    'images':images,
                    'spider_date':spider_date}

#插入Mongodb数据库中
def save_to_mongo(result):
  if result:
    #数据去重操作--当出现之前爬取的图片链接则认为是重复数据
    if db[mongo_table].update({'url':result.get('url')},{'$set':result},True):
        print('插入到MongDB数据库成功')
        return True
  return result

def main():
 #offset,keyword作为可变参数动态传值
 for i in range(2,17):
    html = get_page_index(i*20,'街拍')
    for url in parse_page_index(html):
        if url:
         html = get_page_detail(url)
         if html:
            result = parse_page_detail(html,url)
            save_to_mongo(result)


if __name__ == '__main__':

    main()