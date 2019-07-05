#coding=utf-8

#利用它可以不用正则表达式即可以爬取网页数据
# BeautifulSoup(markup,'html.parser') 执行速度适中、文档容错能力强
# BeautifulSoup(markup,'lxml') 执行速度快、文档容错能力强
# BeautifulSoup(markup,'xml') 执行速度快、唯一一个支持XML的解析器
# BeautifulSoup(markup,'html5lib') 最好的容错性，以浏览器的方式解析文档，生成HTML5格式的文档
from bs4 import BeautifulSoup
import re
import sys

html = '''
<html xmlns=http://www.w3.org/1999/xhtml>
<head>   
<meta http-equiv=Content-Type content="text/html;charset=utf-8">
<meta http-equiv=X-UA-Compatible content="IE=edge,chrome=1">
<meta content=always name=referrer> 
<link rel="shortcut icon" href=/favicon.ico type=image/x-icon> 
<link rel=icon sizes=any mask href=//www.baidu.com/img/baidu_85beaf5496f291521eb75ba38eacbd87.svg>
<title>百度一下，你就知道 </title>
<p class="s-skin-lm s-isindex-wrap1">
<a href=http://top.baidu.com/?fr=mhd_card class=_blank1>实时热点1</a>
<a href=http://top.baidu.com/?fr=mhd_card class=_blank2>实时热点2</a>
</p>
<p class="s-skin-lm s-isindex-wrap2">p2</p>
<p class="s-skin-lm s-isindex-wrap3">p3</p>
<a href=http://top.baidu.com/?fr=mhd_card target=_blank3>实时热点3</a>
<a href=http://top.baidu.com/?fr=mhd_card target=_blank4>实时热点4</a>
<a href=http://top.baidu.com/?fr=mhd_card target=_blank5>实时热点5</a> 
'''

#提取title中的信息

'''
soup = BeautifulSoup(html,'lxml')
#print(soup.prettify(encoding='utf-8')) #因为html是不全的所以prettify是补全完整
print(soup.title.string) 
print(soup.p.a.string)
'''

#标签选择器----获取元素
'''
soup = BeautifulSoup(html,'lxml')
print(soup.title)
print(soup.head)
print(soup.p) #输出第一个匹配结果

'''

#标签选择器---获取名称

'''
soup = BeautifulSoup(html,'lxml')
print(soup.title.name) #title

'''

#标签选择器---获取属性

'''soup = BeautifulSoup(html,'lxml')
print(''.join(soup.a.attrs['class']))
if ''.join(soup.a.attrs['class']) == '_blank1':
    print('找到了对应属性')
else:
    print('未找到对应属性')
print(''.join(soup.a['class'])) #['s-skin-lm', 's-isindex-wrap1']可以获取class属性'''
#标签选择器----嵌套选择


hm = '''
<html xmlns=http://www.w3.org/1999/xhtml>
<head>
   <title>百度一下，你就知道 </title>
</head>
<body>
  <p class="wrap1">
    <a href=http://top.baidu.com/?fr=mhd_card target=_blank1>实时热点1</a>
      输出1
    <a href=http://top.baidu.com/?fr=mhd_card target=_blank2>实时热点2</a>
       输出2
    <a href=http://top.baidu.com/?fr=mhd_card target=_blank3>实时热点3</a>
       输出3
  </p> 
  <p class="wrap2">
    <a href=http://top.baidu.com/?fr=mhd_card target=_blank4>实时热点4</a>
    <a href=http://top.baidu.com/?fr=mhd_card target=_blank5>实时热点3</a>
    <a href=http://top.baidu.com/?fr=mhd_card target=_blank6>实时热点6</a>
  </p>
'''

#打印出子辈的节点数据 children
'''
soup = BeautifulSoup(hm,'lxml')
for i,child in enumerate(soup.p.children): #打印出子辈的节点数据
    print(i,child)
'''

#打印出子孙辈的节点数据 descendants

'''
soup = BeautifulSoup(hm,'lxml')

print(type(soup.p.descendants))
for i,child in enumerate(soup.p.descendants): #打印出子孙辈的节点数据
    print(i,child)
'''

#遍历兄弟节点 next_siblings、previous_siblings
'''
soup = BeautifulSoup(hm,'lxml')
print(list(enumerate(soup.a.next_siblings))) #寻找兄弟节点当前位置向下遍历
print(type(soup.a.next_siblings))
print(list(enumerate(soup.a.previous_siblings))) #寻找兄弟节点当前位置向上遍历
'''


#标准选择器 find_all讲解

'''
soup = BeautifulSoup(hm,'lxml')
#d = soup.find_all(class_ = 'wrap1')
for a in soup.find_all('p'):
  if ''.join(a.attrs['class']) == 'wrap1':
      for b in a.find_all('a'):
         print(b.string) #从多个p标签中找出指定的class属性的标签，再对P标签下的li标签进行遍历
'''



'''
soup = BeautifulSoup(hm,'lxml')
print(soup.find_all(class_ = 'wrap1'))
print(type(soup.find_all(class_ = 'wrap1')))
print(soup.find_all(target = '_blank1')) #指定标签打印
print(type(soup.find_all(target = '_blank1')))
print(soup.find_all(text = '实时热点3'))  #根据文本打印输出
'''

#对于单个网页进行实例讲解
html_test = '''
<ul class="zdsnewslist">

                    <li class="clearfix">
                        <span>2018-05-23</span>
                        <a href="./201805/t20180524_11933253.htm" target="_blank">市金融办关于2016年深圳市股权投资机构营业收入与企业所得财政贡献奖励金融发展专项资金的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2018-05-17</span>
                        <a href="./201805/t20180517_11918724.htm" target="_blank">关于拨付中国平安财产保险股份有限公司金融用地项目建设奖励金的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2018-05-09</span>
                        <a href="./201805/t20180509_11818488.htm" target="_blank">关于人民银行深圳市中心支行FATF国际研讨会费用补助的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2018-05-03</span>
                        <a href="./201805/t20180504_11810294.htm" target="_blank">市金融办关于集中受理2017-2018年度《深圳市扶持金融业发展若干措施》第一批资助项目申报的通知</a>
                    </li>

                    <li class="clearfix">
                        <span>2018-05-03</span>
                        <a href="./201805/t20180504_11810274.htm" target="_blank">市金融办关于集中受理2017-2018年度《深圳市扶持金融业发展若干措施》第一批资助项目申报的通知</a>
                    </li>

                    <li class="clearfix">
                        <span>2018-03-07</span>
                        <a href="./201803/t20180307_10964199.htm" target="_blank">关于深圳财通证券股份有限公司等19家机构申请金融发展专项资金奖励的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2018-01-10</span>
                        <a href="./201801/t20180109_10646061.htm" target="_blank">关于深圳市74家金融机构2016年租房补贴的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2018-01-10</span>
                        <a href="./201801/t20180109_10646057.htm" target="_blank">关于补贴第六届中国（广州）国际金融交易博览会参展经费的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2017-12-22</span>
                        <a href="./201712/t20171222_10622165.htm" target="_blank">关于前海期货有限公司申请金融发展专项资金奖励的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2017-12-04</span>
                        <a href="./201712/t20171204_10102200.htm" target="_blank">关于2017年第二批深圳市中小微企业贷款风险补偿金的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2017-11-29</span>
                        <a href="./201711/t20171128_10038137.htm" target="_blank">市金融办关于2016年深圳市股权投资机构租房补贴金融发展专项资金的公示</a>
                    </li>

                    <li class="clearfix">
                        <span>2017-11-27</span>
                        <a href="./201711/t20171127_10033058.htm" target="_blank">关于2014年下半年2015年度深圳金融发展专项资金资助项目的公示</a>
                    </li>

</ul>

'''

# 打印出每个li目录下的日期、a标签下中文内容、一个完整的a标签
# <a href="./201711/t20171127_10033058.htm" target="_blank">关于2014年下半年2015年度深圳金融发展专项资金资助项目的公示</a>
soup = BeautifulSoup(html_test, 'lxml')
li = soup.find_all(class_='zdsnewslist')  # 查询class属性为zdsnewslist的li标签
for lis in li:
    aa = lis.find_all(class_='clearfix')
    for s in aa:
        title = str(s.a.string) #内容信息
        href = 'http://aaa'+ str(s.a.attrs['href'])[1:]  #取出链接信息
        a1 = '<a href="%s" target="_blank">%s</a>' % (href,title)
        print(title)



html1 = '''
<div class="inNewsList">
            	<ul class="ftdt-list">
					<li><a href="./201808/t20180808_13840785.htm" title="2018年1-7月深圳市社会保险基金预算执行情况表"><em>2018-08-08</em><span>2018年1-7月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					 
					<li><a href="./201807/t20180714_13681601.htm" title="2018年1-6月深圳市社会保险基金预算执行情况表"><em>2018-07-14</em><span>2018年1-6月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201806/t20180611_12118038.htm" title="2018年1-5月深圳市社会保险基金预算执行情况表"><em>2018-06-11</em><span>2018年1-5月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201805/t20180510_11845421.htm" title="2018年1-4月深圳市社会保险基金预算执行情况表"><em>2018-05-10</em><span>2018年1-4月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201804/t20180412_11742206.htm" title="2018年1-3月深圳市社会保险基金预算执行情况表"><em>2018-04-12</em><span>2018年1-3月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201803/t20180313_11173954.htm" title="2018年1-2月深圳市社会保险基金预算执行情况表"><em>2018-03-13</em><span>2018年1-2月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201802/t20180226_10798071.htm" title="2018年1月深圳市社会保险基金预算执行情况表"><em>2018-02-14</em><span>2018年1月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201801/t20180118_10676235.htm" title="2017年1-12月深圳市社会保险基金预算执行情况表"><em>2018-01-15</em><span>2017年1-12月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201712/t20171212_10467071.htm" title="2017年1-11月深圳市社会保险基金预算执行情况表"><em>2017-12-12</em><span>2017年1-11月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201711/t20171108_9559211.htm" title="2017年1-10月深圳市社会保险基金预算执行情况表"><em>2017-11-08</em><span>2017年1-10月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201710/t20171016_9380950.htm" title="2017年1-9月深圳市社会保险基金预算执行情况表"><em>2017-10-16</em><span>2017年1-9月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201709/t20170911_8664187.htm" title="2017年1-8月深圳市社会保险基金预算执行情况表"><em>2017-09-11</em><span>2017年1-8月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201708/t20170809_8073031.htm" title="2017年1-7月深圳市社会保险基金预算执行情况表"><em>2017-08-09</em><span>2017年1-7月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201707/t20170714_7872562.htm" title="2017年1-6月深圳市社会保险基金预算执行情况表"><em>2017-07-14</em><span>2017年1-6月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201706/t20170612_6991756.htm" title="2017年1-5月深圳市社会保险基金预算执行情况表"><em>2017-06-12</em><span>2017年1-5月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201705/t20170512_6692478.htm" title="2017年1-4月深圳市社会保险基金预算执行情况表"><em>2017-05-12</em><span>2017年1-4月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201704/t20170412_6128858.htm" title="2017年1-3月深圳市社会保险基金预算执行情况表"><em>2017-04-12</em><span>2017年1-3月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201703/t20170309_6033654.htm" title="2017年1-2月深圳市社会保险基金预算执行情况表"><em>2017-03-09</em><span>2017年1-2月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201702/t20170214_5964581.htm" title="2017年1月深圳市社会保险基金预算执行情况表"><em>2017-02-14</em><span>2017年1月深圳市社会保险基金预算执行情况表</span></a></li>
					 
					<li><a href="./201701/t20170111_5879785.htm" title="2016年1-12月深圳市社会保险基金预算执行情况表"><em>2017-01-11</em><span>2016年1-12月深圳市社会保险基金预算执行情况表</span></a></li>
					 
                </ul>
            </div>
'''

'''
soup = BeautifulSoup(html1,'lxml')
urls = soup.find_all(class_ = 'ftdt-list')
for url in urls:
    aa =  str(url.a.string)
    print(aa)
'''


#通过css元素提取数据

'''soup = BeautifulSoup(html_test, 'lxml')

for li in soup.select('.zdsnewslist li span'):
     print(li)
     print(type(li))
     span = li.get_text()
     print(span)'''

#通过css元素获取信息


hm3 = '''
<html xmlns=http://www.w3.org/1999/xhtml>
<head>
   <title>百度一下，你就知道 </title>
</head>
<body>
  <ul class="wrap1">
    <li class='element' href=http://top.baidu.com/?fr=mhd_card target=_blank1>实时热点1</li>
      输出1
    <li href=http://top.baidu.com/?fr=mhd_card target=_blank2>实时热点2</li>
       输出2
    <li href=http://top.baidu.com/?fr=mhd_card target=_blank3>实时热点3</li>
       输出3
  </ul> 
  <ul class="wrap2">
    <li href=http://top.baidu.com/?fr=mhd_card id=_blank4>实时热点4</li>
    <li href=http://top.baidu.com/?fr=mhd_card id=_blank5>实时热点5</li>
    <li href=http://top.baidu.com/?fr=mhd_card id=_blank6>实时热点6</li>
  </ul>
   <p class= 'element1'></p>
'''

'''
soup = BeautifulSoup(hm3,'lxml')
for target in soup.select('.wrap2 li'):
    print(target['id']) #获取属性名称
'''

'''soup = BeautifulSoup(hm3,'lxml')

for target in soup.select('.wrap2 li'):
    print(target)
    print(type(target))
    print(target.get_text()) #获取内容'''



