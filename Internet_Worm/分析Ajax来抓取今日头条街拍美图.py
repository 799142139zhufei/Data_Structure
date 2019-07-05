#coding=utf-8

import requests
from urllib.parse import urlencode
from requests.exceptions import RequestException
import json
from bs4 import BeautifulSoup
import pymongo
import re
import os
from hashlib import md5


mongo_url = 'localhost'
mongo_port = 27017
mongo_db = 'toutiao'
mongo_table = 'toutiao'

client = pymongo.MongoClient(mongo_url,mongo_port)
db = client[mongo_db]

def get_page_index(offset,keyword):
        #json的请求参数
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

def parse_page_index(html):

    data = json.loads(html) #将字符串转换成json对象
    if data and 'data' in data.keys():  #判断返回数据中包含data属性；data.keys()返回的是所有的键名
        for item in data.get('data'):
            yield item.get('article_url') #生成一个生成器

def get_page_detail(url):
    print(url)
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

def parse_page_detail(html,url):
    soup = BeautifulSoup(html,'lxml')
    title = soup.select('title')[0].get_text()
    restult = re.search('JSON.parse\("(.*?)"\),',html,re.S)
    if restult:#需要加个判断如果restult为空这返回空值便于后期分析
        data = json.loads(restult.group(1).replace('\\"','"'))  #将字符串转换成json对象
        if data and 'sub_images'in data.keys(): #判断data的所有key中是否包含sub_images
            sub_images = data.get('sub_images')
            images = [image.get('url').replace('\\/','/') for image in sub_images]
            for image in images:
                download_images(image)
            return {
                    'url': url,
                    'title':title,
                    'images':images
            }
    return {
                    'url': url,
                    'title':'',
                    'images':''
            }
def save_to_mongo(result): #图片路径存入mongodb数据库中
    if result:
        if db[mongo_table].insert(result):
            print('MongDB数据库插入成功')
            return True

    return result

def download_images(url): #将图片下载下来
    try:
        hearder = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
        }
        response = requests.get(url,headers = hearder)
        if response.status_code == 200:
            response.encoding = 'utf-8'
            save_image(response.content)
        return None
    except RequestException:
        print('请求图片失败',url)
        return None

def save_image(content):
    # os.getcwd()打印到项目目录；md5(content).hexdigest()避免生成的图片名不重复
    file_path = '{0}/{1}.{2}'.format('D:/image/toutiao_image',md5(content).hexdigest(),'jpg')
    if not os.path.exists(file_path):
        with open (file_path,'wb') as f:
              f.write(content)
              f.close()

def main():
    #offset,keyword作为可变参数动态传值
    html = get_page_index(10,'街拍')
    for url in parse_page_index(html):
        if url:
         html = get_page_detail(url)
         if html:
            result = parse_page_detail(html,url)
            save_to_mongo(result)
if __name__ == '__main__':

    main()