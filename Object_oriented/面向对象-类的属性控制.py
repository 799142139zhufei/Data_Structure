class Student(object):

    def __init__(self,name,age):
        self.name = name
        self.age = age

    def __getattribute__(self, item):
        return super(Student,self).__getattribute__(item)

    def __setattr__(self, key, value):
        self.__dict__[key] = value


if __name__ == '__main__':
    p = Student('zhangsan',12)
    print(p.name)