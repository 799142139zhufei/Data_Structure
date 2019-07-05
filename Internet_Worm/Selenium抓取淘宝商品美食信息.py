#coding=utf-8

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from  pyquery import PyQuery as pq
import pymongo

#SERVICE_ARGS = ['--load-images=false','--disk-cache=true']
#driver = webdriver.PhantomJS(service_args= SERVICE_ARGS)

driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)  # 由于加载页面是需要时间等待故引入时间等待；

mongo_url = 'localhost'
mongo_port = 27017
mongo_db = 'taobao'
mongo_table = 'taobao'

client = pymongo.MongoClient(mongo_url,mongo_port)
db = client[mongo_db]

def search():

    try:
        #进入到淘宝首页
        driver.get('https:///www.taobao.com')
        #模拟了一个输入框在input框输入对应的关键字
        input = wait.until(
            # By.CSS_SELECTOR指定选择器的类型、#q输入框的selector；也可以选择其它选择器
            EC.presence_of_element_located((By.CSS_SELECTOR, '#q'))
        )
        #模拟了一个提交按钮查询对应的信息（注意在这里可以灵活使用不同的选择器有时候报错就是选择的选择器问题导致）
        submit = wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="J_TSearchForm"]/div[1]/button')))
        #模拟输入内容后、再提交信息
        input.send_keys('美食')
        submit.click()
        get_products() #调用程序
        #获取返回的总页数
        totile = wait.until(EC.presence_of_element_located((By.XPATH,'//*[@id="mainsrp-pager"]/div/div/div/div[1]')))
        return totile.text
    except TimeoutError: #当请求时间过长时再次请求即可
        return search()

#循环遍历加载浏览器
def next_page(page_number):

  try:
    #模拟输入指定的页数
    input = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsrp-pager > div > div > div > div.form > input'))
    )
    #确认提交按钮
    submit = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="mainsrp-pager"]/div/div/div/div[2]/span[3]')))
    input.clear() #清除input框中的信息
    input.send_keys(page_number)
    submit.click()
    #用于判断输入的页数和跳转的页数是否相等
    wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR,'#mainsrp-pager > div > div > div > ul > li.item.active > span'),str(page_number)))
    get_products() #判断成功调用
  except TimeoutError:
      return next_page(page_number)


#解析浏览器页面
def get_products():
    #判断是否加载完成
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsrp-itemlist .items .item')))
    html = driver.page_source #获取整个网页的源代码
    doc = pq(html)
    items = doc('#mainsrp-itemlist .items .item').items() #获取所有循环的内容
    for item in items:
        product = {
            'image':item.find('.pic .img').attr('src'),
            'price':item.find('.price').text(),
            'deal':item.find('.deal-cnt').text()[0:-3],
            'title':item.find('.title').text(),
            'shop':item.find('.shop').text(),
            'location':item.find('.location').text()
        }
        mongo_insert(product)

# 往Mongo数据库插入数据
def mongo_insert(product):
    print(product)
    try:
        if db[mongo_table].insert(product):
            print('Insert MongoDB数据库成功')
    except Exception:
        print('Insert MongoDB数据库失败')

def main():

    totile = search()
    sum = int(re.search('(\d+)',totile,re.S).group(1)) #打印出总页数
    for i in range(2,sum + 1):
        next_page(i)
    driver.close() #待main方法执行完成后关闭浏览器

if __name__ == '__main__':

    main()