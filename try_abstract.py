#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:10:54 2019

@author: li-ming-fan
"""

'''
1、如果子类在继承后一定要实现的方法，可以在父类中指定metaclass为abc模块的ABCMeta
类，并在指定的方法上标准abc模块的@abcstractmethod来达到目的。
2、一旦定义了这样的父类，父类就不能实例化了，否则会抛出TypeError异常。
3、继承的子类如果没有实现@abcstractmethod标注的方法，在实例化使也会抛出TypeError异常。
'''

from abc import ABCMeta, abstractmethod
 
class Tester(metaclass=ABCMeta):
    def __init__(self,name,rank,salary):
        self.name = name
        self.rank = rank
        self.salary =salary
 
    def __str__(self):
        return "('{name}','{rank}',{salary})".format(**vars(self))
 
    @abstractmethod
    def test(self):
        pass
 
class FunctionTester(Tester):
    def test(self):
        print("功能测试")
 
class AutoTester(Tester):
    def test(self):
        print("自动化测试")
 
class PerformanceTester(Tester):
    def notest(self):
        print("因为没有定义test方法，故实例化会报错")
 
 
if __name__ == "__main__":
    test1 = FunctionTester("cxa","初级测试工程师",6000)
    test1.test()
    test2 = AutoTester("cxb","中级测试工程师",10000)
    test2.test()
    try:
        test3 = Tester("cxc","测试工程师",20000) #会抛出TypeError异常
    except TypeError as e:
        print("因为定义了抽象方法，故不能实例化",e,sep="\n")
    try:
        test4 = PerformanceTester("cxd","高级测试工程师",30000) #会抛出TypeError异常
    except TypeError as e:
        print("因为没有定义test方法，故实例化会报错",e,sep="\n")
        
        