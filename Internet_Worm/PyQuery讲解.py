#coding=utf-8

from pyquery import  PyQuery as pq

html = '''
    <li href=http://top.baidu.com/?fr=mhd_card class=_blank1>实时热点1</li>
    <li href=http://top.baidu.com/?fr=mhd_card class=_blank2>实时热点2</li>
    <li href=http://top.baidu.com/?fr=mhd_card class=_blank3>实时热点3</li>
    <li href=http://top.baidu.com/?fr=mhd_card class=_blank4>实时热点4</li>
    <li href=http://top.baidu.com/?fr=mhd_card class=_blank5>实时热点5</li>
    <li href=http://top.baidu.com/?fr=mhd_card class=_blank6>实时热点6</li>
'''

'''
doc = pq(html) #声明成pyquery对象
print(doc('li'))

'''

'''
doc = pq(url='http://www.baidu.com') #传入网页链接后抓取内容再进行解析
print(doc('head'))

'''

'''
doc = pq(filename='2.html') #解析文件
print(doc('li'))

'''

html1 = '''
<div id = "contain">
  <ul class = "list">
    <ul class = "list1">
      <li href="http://top.baidu.com/?fr=mhd_card" class="_blank1">实时热点1</li>
    </ul>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank2 active">实时热点2</li>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank2">实时热点3</li>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank4">实时热点4</li>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank5 active">实时热点5</li>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank6">实时热点6</li>
   </ul>
</div>
'''

#基于CSS选择器

'''
doc = pq(html1)
print(doc('#contain  .list li')) 

'''

#查找子元素
'''
doc = pq(html1)
items = doc('.list')
lis = items.find('._blank1')
print(lis)

'''


#爬取它子元素第一级元素
'''
doc = pq(html1)
items = doc('.list')
lis = items.children()
print(lis)
'''

# parents查找出对应的所有外节点；parent找出对应的第一个父节点
'''
doc = pq(html1)
items = doc('._blank1')
lis = items.parents('.list')  #找出指定的父元素
print(lis)

'''

'''
doc = pq(html1)
items = doc('.list ._blank2.active') #寻找样式为class= _blank2 active
print(items.siblings()) #输出与之平级的元素
print(items.siblings('.active')) #输出兄弟中包含active的元素
'''

########################################元素的遍历#####################################################
html2 = '''
<div id = "contain">
  <ul class = "list">
    <a href="http://top.baidu.com/?fr=mhd_card" class="_blank2 active">实时热点2</a>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank2">实时热点3</li>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank4"><a href = "http://www.baidu.com">输出结果了</a></li>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank5 active">实时热点5</li>
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank6">实时热点6</li>
   </ul>
</div>
'''


doc = pq(html2)
lis = doc('#contain .list').attr.href #通过items方式进行循环遍历
print(lis)
for li in lis:
    print(li.attr.href)



#获取信息、属性

'''doc = pq(html2)
a = doc('._blank4 a')
print(a)
print(a.text()) #输出结果了
print(a.attr('href')) #http://www.baidu.com
print(a.attr.href) #http://www.baidu.com'''


############################DOM操作 addClass、removeClass#############################
'''
doc = pq(html2)
a = doc('._blank2.active')
print(a)
a.remove_class('active')
print(a)
a.add_class('active')
print(a)

'''
############################attr、css修改元素属性#############################

'''doc = pq(html2)
a = doc('._blank2.active')
print(a)
a.attr('name','link') #如果存在name、font-size属性则更新原来的值，如果不存在则追加
a.css('font-size','14px')
print(a)'''

html3 = '''
<div id = "contain">
    输出结果了
    <li href="http://top.baidu.com/?fr=mhd_card" class="_blank2 active">实时热点2</li>
</div>
'''

doc = pq(html3)
a = doc('#contain')
print(a.text())
b= a.find('li') #删除掉指定标签，只剩下对应的文本信息输出
print(b.attr.href)


############################伪类选择器#############################
'''doc = pq(html2)
lis = doc('.list')
li = lis('li:first-child') #获取第一个li标签
print(li)
li = lis('li:last-child') #获取最后一个li标签
li = lis('li:nth-child(2)') #获取指定li标签：第2个标签
li = lis('li:gt-child(2)') #获取2个后面的li标签
li = lis('li:nth-child(2n)') #获取偶数的li标签
li = lis('li:contains(second)') #获取包含second文本信息的li标签'''








