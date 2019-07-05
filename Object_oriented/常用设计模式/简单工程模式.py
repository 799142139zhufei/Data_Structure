
'''不直接向客户暴露对象创建的实现细节，
而是通过一个工厂来负责创建产品类的实例'''

class Payment(object):
    '''抽象产品角色'''
    def pay1(self):
        print('我是基类')

class PayTest1(Payment):
    '''具体产品角色'''
    def pay(self):
        print('我是测试1')
        super(PayTest1,self).pay1() # 子类继承了父类方法

class PayTest2(Payment):
    '''具体产品角色'''
    def pay(self):
        print('我是测试2')

class PayFactory(object):
    '''工厂角色'''
    def create(self,pay):
         if pay == 'PayTest1':
           return PayTest1()
         if pay == 'PayTest2':
            return PayTest2()
         else:
             None

# 所有的类和方法统一在一个入口调用，代码复用性好
pf = PayFactory()
obj = pf.create('PayTest2')
obj.pay()
obj.pay1()