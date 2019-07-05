#coding=utf-8

import requests
from requests.exceptions import ConnectTimeout,ReadTimeout,HTTPError,RequestException
import urllib.request
import urllib.parse
import socket
import urllib.error
#如何网页爬取图片
'''
response = requests.get('https://ss0.bdstatic.com/5aV1bjqh_Q23odCf/static/mantpl/img/base/loading_72b1da62.gif')
print(response.content) #获取二进制代码
with open ('./1.gif','wb') as f:
      f.write(response.content)
      f.close();
'''

#打印输出页面源码

from selenium import webdriver
driver = webdriver.PhantomJS()
#driver = webdriver.Chrome()
driver.get('http://www.baidu.com')
print(driver.page_source)


#基于urllib进行页面爬取 get请求方式
'''
url = 'http://www.baidu.com/s?wd='
key = 'zhufei的博客'
key_code = urllib.request.quote(key) ##因为URL里含中文，需要进行编码
url_key = url+key_code
header={
    'User-Agent':'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

requests = urllib.request.Request(url_key,headers= header)
response = urllib.request.urlopen(requests).read()

with open ('./baidu.html','wb') as f:
     f.write(response)
     f.close()
'''

#基于urllib进行页面爬取 post请求方式
'''
url='http://www.iqianyue.com/mypost'
header={
   'User-Agent':'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

data  = {
    'name' :'aa',
    'pass' : 'bb'
}
posdata = urllib.parse.urlencode(data).encode('utf-8')
request=urllib.request.Request(url,posdata,headers=header)
reponse=urllib.request.urlopen(request).read()

fhandle=open("./2.html","wb")
fhandle.write(reponse)
fhandle.close()

'''

#打印响应超时 http://httpbin.org/get
'''
try:
    reponse = urllib.request.urlopen('http://www.baidu.com',timeout=0.01)
except urllib.error.URLError as e:
    if isinstance(e.reason,socket.timeout):
        print('Time Out')
'''


#利用python进行文件上传操作
'''
{  
  "field1" : open("filePath1", "rb")),  
  "field2" : open("filePath2", "rb")),  
  "field3" : open("filePath3", "rb"))  
}  
files = {'file':open('1.gif','rb')}
response = requests.post('http://httpbin.org/post',files = files)
print(response.text)

'''

#获取cookie
'''
response = requests.get('http://www.baidu.com')
print(response.cookies)
for key,value in response.cookies.items():
    print(key+ '=' +value)
'''


'''
response = requests.get('http://www.12306.cn',verify = False) #消除SSL
print(response.text)

'''

#通过requests判断异常处理设置
'''
try:
   response = requests.get('http://www.baidu.com',timeout = 0.01)
   print('aa')
except ConnectTimeout:
   print('bb')
except RequestException:
    print('cc')
except HTTPError:
    print('dd')
except ReadTimeout:
    print('ff')
'''


#通过requests 进行认证设置（在登录一个网站时需要输入用户名和密码）
'''
response = requests.get('httt://www.baidu.com',auth = {'user','123'})
print(response.status_code)
'''


