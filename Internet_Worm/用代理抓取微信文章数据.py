#coding=utf-8

'''
流程框架：
1、抓取索引页内容：利用Requests请求目标站点，得到索引网页HTML代码，返回结果。
2、代理设置：遇到状态码为302时，则证明IP被封切换代理重试。（重点）
3、分析详情页内容：请求详情页，分析得到标题，正文等内容。（多种解析方式：正则表达式、bs4、pyquery等技术）
4、数据入库：将结构化数据插入到MongoDB数据库
'''

import requests
from urllib.parse import urlencode
from requests.exceptions import ConnectionError
import redis

proxy_pool_url = 'http://localhost:5000/get' #动态获取ip
max_count = 5 #判断请求链接次数达到5次后弹出

proxy = None # 最开始使用本机进行爬取，当出现302后再使用代理进行爬取

header = {
    'Cookie': 'SUV=1529315847376662; SMYUV=1529315847378025; UM_distinctid=1641253ecd8371-04e06effbfb8af-47e1137-e1000-1641253ecd9917;'
              ' IPLOC=CN4403; SUID=1F2CE9657C20940A000000005B2DC68E; ABTEST=0|1529726608|v1; '
              'weixinIndexVisited=1; sct=1; LSTMV=164%2C27; LCLKINT=2610; SUIR=B68541CDA8ADC7EE93BD16FBA925C11C;'
              ' SNUID=2013A95A3F45500ACDE9AB9B40E0255A; JSESSIONID=aaa7SJE1QY2iC_iAR9lnw',
    'Host': 'weixin.sogou.com',
    'Referer': 'https://weixin.sogou.com/',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36'
         }

#获取代理值
def get_proxy():
    try:
        response = requests.get(proxy_pool_url)
        if response.status_code == 200:
            return response.text
        return None
    except ConnectionError:
        return None

def get_html(url,count = 1):
    global proxy #引用全局变量
    if count >= max_count:
        print('请求次数过多')
        return None
    try:
            if proxy:
                proxies = {
                   'http' : 'http://' + proxy
                }
                print('输出代理：',proxies)
                response = requests.get(url, allow_redirects=False,headers = header,proxies = proxies)  # 避免自动跳转存在代理时
            else:
                response = requests.get(url, allow_redirects=False,headers = header)  # 避免自动跳转不存在代理时
            if response.status_code == 200:
                response.encoding = 'utf-8'
                return response.text
            if response.status_code == 302:
                # 说明IP已经被封了 Need Proxy
                print('IP已经被封了！')
                proxy = get_proxy()  # 获取代理
                if proxy:
                    print('Using Proxy', proxy)
                    #count += 1
                    return get_html(url)
                else:
                    print('Get Proxy Fail')
                    return None
    except ConnectionError:
        count += 1
        print('请求链接失败')
        return get_html(url,count)


# Requests请求获取网页信息
def get_index(keyword, page):

    data = {
        'query': keyword,
        'type': 2,
        'page': page,
        'ie': 'utf8'
    }
    request_url = 'http://weixin.sogou.com/weixin?' + urlencode(data)  # 将字典类型转成链接参数
    html = get_html(request_url)
    return html



def main():
    for page in range(1,101):
        html =  get_index('风景',page)
        print(html)


if __name__ == '__main__':

    main()