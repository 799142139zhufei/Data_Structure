'''
定义:定义一个工厂类接口,让工厂子类来创建一系列相关或相互依赖的对象
角色:抽象工厂角色,具体工厂角色,抽象产品角色,具体产品角色,客户端
适用场景:系统要独立于产品的创建和组合时,强调一系列相关产品的对象设计以便进行联合调试时,提供一个产品类库,想隐藏产品的具体实现时
优点:将客户端与类的具体实现相分离,每个工厂创建了一个完整的产品系列,易于交换产品.有利于产品的一致性
缺点:难以支持新种类的产品

对于校区有：北京校区和上海校区（产品族）
每个校区下的课程有：python、java和c++（产品等级）
抽象工厂是基于产品族建立工厂这样便于减少内存和硬盘消耗；
'''


class AbstractCpu(object):
    '''产品等级抽象类'''
    def __init__(self,name,price):
        self.name = name
        self.price = price

    def Cpu(self):
        print('该款%s价格%s元'%(self.name,self.price))

class AmdCpu(AbstractCpu):
    '''具体产品衍生类'''
    def __init__(self,name,preice,pl):
        super(AmdCpu,self).__init__(name,preice)
        self.pl = pl

    def Amd_Cpu(self):
        print('该款%s价格%s元评论数%s' % (self.name, self.price,self.pl))

class IntelCpu(AbstractCpu):
    '''具体产品衍生类'''

    def __init__(self, name, preice, pl):
        super(IntelCpu, self).__init__(name, preice)
        self.pl = pl

    def Intel_Cpu(self):
        print('该款%s价格%s元评论数%s' % (self.name, self.price,self.pl))


class AbstractMianboard(object):
    '''产品等级抽象类'''
    def __init__(self,name,price):
        self.name = name
        self.price = price

    def Mianboard(self):
        print('该款%s价格%s元'%(self.name,self.price))


class AmdMianboard(AbstractMianboard):
    '''具体产品衍生类'''
    def __init__(self,name,preice,pl):
        super(AmdMianboard,self).__init__(name,preice)
        self.pl = pl

    def Amd_Mianboard(self):
        print('该款%s价格%s元评论数%s' % (self.name, self.price,self.pl))

class IntelAmdMianboard(AbstractMianboard):
    '''具体产品衍生类'''
    def __init__(self, name, preice, pl):
        super(IntelAmdMianboard, self).__init__(name, preice)
        self.pl = pl

    def Intel_Mianboard(self):
        print('该款%s价格%s元评论数%s' % (self.name, self.price,self.pl))

class AbstractFactory(object):
    '''工厂抽象类'''
    def Cpu(self):
        pass

    def mianboard(self):
        pass


class AmdFactory(AbstractFactory):
    '''具体工厂'''
    def createCpu(self):
        return AmdCpu(None,None,None)

    def createmianboard(self):
        return AmdMianboard(None,None,None)


class IntelFactory(AbstractFactory):
    '''具体工厂'''
    def createCpu(self):
        return IntelCpu(None,None,None)

    def createmianboard(self):
        return IntelAmdMianboard(None,None,None)


class ComputerEngineer(object):
    '''客户端'''
    def prepareHardwraes(self,compter_factory):  # 通过判断不同的产品族来调用产品
        self.cpu = compter_factory.createCpu() # 需要组装什么配置的cpu
        self.mianboard = compter_factory.createmianboard() # 需要组装什么配置的主板

        return self.cpu,self.mianboard


if __name__ == '__main__':

    CE = ComputerEngineer()  # 实例化客户端

    AF = AmdFactory() # 实例化工厂对象
    af_cpu,af_mianboard= CE.prepareHardwraes(AF) # 组装产品类

    #外部给该产品CPU赋值
    af_cpu.name = '苹果电脑CPU'
    af_cpu.price = '1000'
    af_cpu.pl = '1000'
    af_cpu.Amd_Cpu()
    #外部给该产品主板赋值
    af_mianboard.name = '苹果电脑主板'
    af_mianboard.price = '2000'
    af_mianboard.pl = '2000'
    af_mianboard.Amd_Mianboard()

    IF = IntelFactory() # 实例化工厂对象
    if_cpu, if_mianboard = CE.prepareHardwraes(IF) # 组装产品类
    #外部给该产品CPU赋值
    if_cpu.name = '华硕电脑CPU'
    if_cpu.price = '1000'
    if_cpu.pl = '1000'
    if_cpu.Intel_Cpu()
    #外部给该产品主板赋值
    if_mianboard.name = '华硕电脑主板'
    if_mianboard.price = '2000'
    if_mianboard.pl = '2000'
    if_mianboard.Intel_Mianboard()


