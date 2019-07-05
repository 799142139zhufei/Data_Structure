#coding=utf-8

import  requests
import re
from multiprocessing import Pool
import json
from requests.exceptions import RequestException

#获取浏览链接并返回网页代码,添加异常捕捉机制防止程序中途停止终端
def get_one_page(url):
  try:
    header = {
         'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
    }
    response = requests.get(url,headers= header) #该网站需要伪造一个身份才能访问
    if response.status_code == 200:
       response.encoding = 'utf-8'
       return response.text
    return None
  except:
      return None

def parse_one_page(html):
    pattern = re.compile('<dd>.*?board-index.*?>(\d+)</i>.*?data-src="(.*?)".*?name">'
                         '<a.*?>(.*?)</a></p>.*?star">(.*?)</p>.*?releasetime">(.*?)'
                         '</p>.*?integer">(.*?)</i>.*?fraction">(.*?)</i></p>.*?</dd>',re.S) #获取则表达式字符
    items = re.findall(pattern,html)  #一次性全部打印出所有的匹配内容
    for item in items:
        yield {
            'index': item[0],
            'image': item[1],
            'title': item[2],
            'actor': item[3].strip(), #去除掉换行符
            'time':  item[4].strip(),
            'score': item[5]+item[6]
        }

def write_to_file(content):
    with open ('result.text','a',encoding='utf-8') as f:
        f.write(json.dumps(content,ensure_ascii=False) + '\n') #需要将字典转换成字符串形式,设置ensure_ascii为False保证输出的中文是正常的格式
        f.close()

def main(offset):
    url = 'http://maoyan.com/board/4?offset='+ str(offset)
    html = get_one_page(url)
    for itme in parse_one_page(html):
        #print(type(itme))
        write_to_file(itme)

if __name__ == '__main__':
    #for i in range(10):
       # main(10)
    pool =  Pool(8)
    result = pool.map(main,[i*10 for i in range(10)]) #爬取的主函数、传入main函数的入参



