#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from pyspider.libs.base_handler import *
import pymongo


class Handler(BaseHandler):
    crawl_config = {

    }
    mongo_url = 'localhost'
    mongo_port = 27017
    mongo_db = 'tripadvisor'
    mongo_table = 'tripadvisor'
    client = pymongo.MongoClient(mongo_url, mongo_port)
    db = client[mongo_db]

    @every(minutes=24 * 60) #每隔一天的请求频率
    def on_start(self):
        self.crawl('https://www.tripadvisor.cn/Attractions-g186338-Activities-London_England.html',
                   callback=self.index_page, validate_cert=False)

    @config(age=10 * 24 * 60 * 60)  #判断该请求是否已经过期，如果没有过期不会重新请求
    def index_page(self, response):
        for each in response.doc('.shelf_row_1 .shelf_title_container > a').items():
            self.crawl(each.attr.href, callback=self.detail_page, validate_cert=False)

        next = response.doc('.unified pagination .nav next').attr.href
        self.crawl(next, callback=self.index_page, validate_cert=False)

    @config(priority=2)
    def detail_page(self, response):

     return {
        'title1': response.doc('.ppr_priv_trip_planner h1').text(),
        'title2': response.doc('.shelf_row_1 .shelf_title_container > a').text(),
        'dianpinig': response.doc('.more > a').text(),
        'url': response.url,
        'title': response.doc('title').text()
       }

    def on_result(self, result):
        if result:
            self.save_mongo(result)

    def save_mongo(self, mongo_table,result):
        if self.db[mongo_table].insert(result):
           print('成功插入数据')















