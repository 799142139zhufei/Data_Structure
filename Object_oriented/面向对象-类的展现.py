class Student(object):

    def __init__(self,name,age):
        self.name = name
        if isinstance(age,int):
            self.age = age
            print('开始运行......')
        else:
            raise Exception ('age is not int type!')

    def __str__(self):
        if isinstance(self.age,int):
            return '%s is %s' %(self.name,self.age)  # 返回一个字符串
        else:
            return  '不是字符串'

if __name__ == '__main__':
    Student1 = Student('zhangsan',25)
    print(Student1)