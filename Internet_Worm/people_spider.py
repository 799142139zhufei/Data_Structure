#coding = utf-8


import requests
from lxml import etree
import pymysql


class PeopleSpider(object):

    def __init__(self):
        self.content = []
        self.base_url = 'http://www.jr.sz.gov.cn/sjrb/xxgk/zjxx/zxzjxx/'
        self.headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:39.0) Gecko/20100101 Firefox/39.0"}

    def send_request(self,url):
        """发送请求，返回相应"""
        html = requests.get(url=url,headers=self.headers)
        html.encoding = 'utf-8'
        return html.text

    def parse_page(self,html):
        """解析页面"""
        selector = etree.HTML(html)
        # 获取所有li标签
        li_list = selector.xpath("//ul[@class='zdsnewslist']/li")

        for li in li_list:
            # 发布日期
            date = li.xpath("./span/text()")[0]
            # 标题
            title = li.xpath("./a/text()")[0]
            # 链接地址
            link = li.xpath("./a/@href")[0]
            # 完整链接地址
            link = self.base_url+link[2:]
            dict1 = {"date":date,"title":title,"link":link}
            self.content.append(dict1)



    def save_data(self):
        """保存数据"""
        conn = pymysql.connect(host='localhost',
                               port=3306,
                               user='root',
                               passwd='123456',
                               db='db',
                               use_unicode=True,
                               charset="utf8")
        cur = conn.cursor()
        for dict1 in self.content:
            sql_str = "insert into peple (title,date,link) VALUE(%s,%s,%s)"
            cur.execute(sql_str,(dict1.get("title"), dict1.get("date"), dict1.get("link"))) #可以防止sql注入
        conn.commit()
        cur.close()
        conn.close()

    def main(self):
        """主函数，对整体进行调度"""
        url = self.base_url
        html = self.send_request(url)
        self.parse_page(html)
        self.save_data()

if __name__ == '__main__':
    title = PeopleSpider()
    title.main()