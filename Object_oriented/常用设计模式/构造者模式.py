'''
定义:将一个复杂对象的构建与它的表示分离,使得同样的构建过程可以创建不同的表示
角色:抽象建造者,具体建造者,指挥者,产品
适用场景:当创建复杂对象的算法应该独立于对象的组成部分以及它的装配方式,当构造过程允许被构造的对象有不同的表示
优点:隐藏了一个产品的内部结构和装配过程,将构造代码与表示代码分开,可以对构造过程进行更精确的控制
'''

from  abc import abstractmethod, ABCMeta

class person(object):
    '''产品'''
    def __init__(self,name = None,age = None,wage = None):
        self.name = name
        self.age = age
        self.wage = wage

    def __str__(self):
        return  '%s今年%d岁工资%d元' %(self.name,self.age,self.wage)

class person_builder(object):
      '''建造者'''
      def build_face(self):
          print('测试。。。。。')

      def build_body(self):
          pass

      def build_arm(self):
          pass

class BeautifulWoman(person_builder):
       '''具体建造者'''
       def __init__(self):
           self.person = person()

       def build_face(self):
           self.person.name = '张三'

       def build_body(self):
           self.person.age = 28

       def build_arm(self):
           self.person.wage = 18000

       def get_person(self):
           return self.person

class persondirecter(object):
    def build_person(self, builder):
        builder.build_face()
        builder.build_body()
        builder.build_arm()
        return builder.get_person()

BeautifulWoman =  BeautifulWoman()
persondirecter =  persondirecter()
p = persondirecter.build_person(BeautifulWoman)
print(p)