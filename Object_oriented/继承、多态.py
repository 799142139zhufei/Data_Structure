class Student(object):
    hobby = '测试'
    def __init__(self, name, age,age1):
        self.__name = name
        self._age = age
        self._age1 = age1

    @classmethod
    def test1(cls):
        return cls.hobby

    @property
    def test2(self):
        return self._age

    def test3(self):
         print(self._age1)

class Student2(object):

    hobby = '测试'
    def __init__(self, name, age,age1):
        self.__name = name
        self._age = age
        self._age1 = age1

    @classmethod
    def test1(cls):
        return cls.hobby

    @property
    def test2(self):
        return self._age

    def test3(self):
         print(self._age1)

class Student1(Student):

    def __init__(self,name, age,age1,age2):
        super(Student1,self).__init__(name, age,age1)
        self._age2 = age2

    def test3(self):
        print(self._age2)


def test(Student1):
    if isinstance(Student1,Student): # 判断Student1对象是不是Student对象或者是它的子类
        Student1.test3()


LiLei = Student('0我是测试','1我是测试','2我是测试')
LiLei1 = Student1('我是测试0','我是测试1','我是测试2','我是测试3')
test(LiLei)
test(LiLei1)