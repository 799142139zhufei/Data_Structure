
'''
定义:定义一个创建对象的接口(工厂接口),让子类决定实例化哪个接口
角色:抽象工厂角色,具体工厂角色,抽象产品角色,具体产品角色
适用场景:需要生产多种,大量复杂对象的时候,需要降低代码耦合度的时候,当系统中的产品类经常需要扩展的时候
优点:每个具体的产品都对应一个具体工厂,不需要修改工厂类的代码,工厂类可以不知道它所创建的具体的类,隐藏了对象创建的实现细节
缺点:每增加一个具体的产品类,就必须增加一个相应的工厂类
'''


class School(object):
      '''抽象产品（学校）'''
      def __init__(self,name,age):
          self.name = name
          self.age = age

      def pay(self):
          print('%s总部成立于%s年' %(self.name,self.age))

class BJ_School(School):
      '''学校衍生类'''
      def __init__(self,name,age,adress):
          super(BJ_School, self).__init__(name, age)
          self.adress = adress

      def create_curriculum(self, curriculum_type): # 创建简单工厂

          if curriculum_type == 'BJ':
              curriculum = BJ_curriculum('java','2019','java')
          elif curriculum_type == 'SH':
              curriculum = SH_curriculum('C++', '2019', 'C++')
          return curriculum

      def BJ_pay(self):
          print('%s总部成立于%s年%s'%(self.name,self.age,self.adress))


class SH_School(School):
    '''学校衍生类'''
    def __init__(self, name, age, adress):
        super(SH_School, self).__init__(name, age)
        self.adress = adress

    def create_curriculum(self,curriculum_type): # 创建简单工厂

        if curriculum_type == 'BJ':
            curriculum = BJ_curriculum('java1', '2019', 'java1')
        elif curriculum_type == 'SH':
            curriculum = BJ_curriculum('C++1', '2019', 'C++1')
        return curriculum

    def SH_pay(self):
        print('%s总部成立于%s年%s' % (self.name, self.age, self.adress))


class curriculum(object):
    '''抽象产品（课程）'''
    def __init__(self, name, time):
        self.name = name
        self.time = time

    def curr(self):
        print('该%s课程成立于%s年' % (self.name, self.time))

class BJ_curriculum(curriculum):
    '''产品衍生类'''
    def __init__(self, name, time,adress):
        super(BJ_curriculum,self).__init__(name, time)
        self.adress = adress

    def BJ_curr(self):
        print('该%s课程成立于%s年%s' % (self.name, self.time,self.adress))


class SH_curriculum(curriculum):
    '''产品衍生类'''
    def __init__(self, name, time, adress):
        super(SH_curriculum, self).__init__(name, time)
        self.adress = adress

    def SH_curr(self):
        print('该%s课程成立于%s年%s' % (self.name, self.time, self.adress))



# 减少实例化对象降低解耦
BJ = BJ_School('北京','2019','北京')
SH = SH_School('上海','2018','上海')

BJ.BJ_pay()
curr1 = BJ.create_curriculum('BJ')
curr1.BJ_curr()

SH.SH_pay()
curr2 = BJ.create_curriculum('SH')
curr2.SH_curr()
