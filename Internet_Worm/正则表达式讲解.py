#coding=utf-8

import re

'''
content = 'Hello 1234 Word_This'
result = re.match('Hello\s(\d+)\sWord_This$',content)
print(result.group(1)) #group中有几个括号就有多少位，指定数字查找对应的数据

'''

#对于$符合进行转义
'''
content = 'price is $5.00 money'
result = re.match('price is \$5\.00 money',content)
print(result.group())

'''

#在能使用search就尽量不要用match
#result = re.match('rice is \$5\.00 money',content) 此时会报错
# result = re.search('rice is \$5\.00 money',content) 就不会报错  区别在于match是从第一个字符串开始查找


content = '''<div class="zx_ml_list_page">
<script>createPageHTML(50, 0, "index","htm","500");</script>
</div>'''
result = re.search('createPageHTML.*?(\d+),',content)
print(result.group(1))


#通过search方法提取数据
'''
content = '<a id="cb_post_title_url" class="postTitle2" singer="学习">Python正则表达式指南</a>'
result = re.search('singer="(.*?)">(.*?)</a>',content,re.S)
if result:
    print(result.group(1)+result.group(2))
'''

#通过sub方法替换数据
'''
content = '<a id="cb_post_title_url" class="postTitle2" singer="学习">Python正则表达式指南</a>'
result = re.sub('(\d+)',r'\1 23',content,re.S) #\1将的作用就是将括号里面的内容拿过来进行了一个替换
print(result)

'''
'''
content = '<a id="cb_post_title_url" class="postTitle2" singer="学习">Python正则表达式指南</a>'
result = re.sub('2','3',content,re.S) #\1将的作用就是将括号里面的内容拿过来进行了一个替换
print(result)
'''
'''
content = '<a id="cb_post_title_url" class="postTitle2" singer="学习">Python正则表达式指南</a>'
result = re.sub('<a.*?>|</a>','',content)
print(result)  #Python正则表达式指南；'|'是依从先左后右在满足左边的条件时则不管右边的匹配结果，
'''


a= '<a href="./201805/t20180504_11810294.htm" target="_blank">市金融办关于集中受理2017-2018年度《深圳市扶持金融业发展若干措施》第一批资助项目申报的通知</a>'

#print(a[11:])