class Student(object):

    def __init__(self,name,age):
        self.name = name
        if isinstance(age,int):
          self.age = age
        else:
            raise Exception('age is not int type')

    def __eq__(self, other):
        if isinstance(other,Student):
            if self.age == other.age:
                return  True
            else:
                return False
        else:
            raise Exception('不是同一个对象！')

    def __add__(self, other):
        if isinstance(other,Student):
            return self.age + other.age
        else:
            return False

if __name__ == '__main__':

    Student1 = Student('a1',10)
    Student2 = Student('a2',20)
    print(Student2 == Student1) # Student2对比时跟顺序存在关系谁在前self.age就是谁的值
    print(Student1 + Student2)