import  threading
'''
定义:保证一个类只有一个实例,并提供一个访问它的全局访问点
适用场景:当一个类只能有一个实例而客户可以从一个众所周知的访问点访问它时
优点:对唯一实例的受控访问,相当于全局变量,但是又可以防止此变量被篡改
'''

# 在实例化Singleton采用多线程时，就会产生两个线程同时执行；会导致
class Singleton(object):
    # 如果该类已经有了一个实例则直接返回,否则创建一个全局唯一的实例
    _instance_lock = threading.Lock()  # 建立一把锁
    def __new__(cls, *args, **kwargs):
          if not hasattr(cls,'_instance'): # 如果不存在则创建实例化对象
              with Singleton._instance_lock: # 使用锁，with内部代码同时只能有一个线程执行
                    cls._instance = super(Singleton,cls).__new__(cls,*args,**kwargs) # 调用该类的父类实例化
          return cls._instance


class Myclass(Singleton): # 继承了Singleton类全局只有一个实例化对象，如果继承object类时可以实例多个对象
    def __init__(self,name):
        if name:
            self.name = name

a = Myclass('张三')
b = Myclass('李四')
print(a.name,b.name)


'''
class Singleton(type):
    # 如果该类已经有了一个实例则直接返回,否则创建一个全局唯一的实例
    _instance_lock = threading.Lock()  # 建立一把锁
    def __call__(cls, *args, **kwargs):
          if not hasattr(cls,'_instance'): # 如果不存在则创建实例化对象
              with Singleton._instance_lock: # 使用锁，with内部代码同时只能有一个线程执行
                     cls._instance = super(Singleton,cls).__call__(*args,**kwargs) # 调用该类的父类实例化
          return cls._instance


class Myclass(metaclass=Singleton): # 继承了Singleton类全局只有一个实例化对象，如果继承object类时可以实例多个对象
    def __init__(self,name):
        if name:
            self.name = name


a = Myclass('张三')
b = Myclass('李四')

print(a.name,b.name)
'''
