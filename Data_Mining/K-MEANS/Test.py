#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import  numpy as np


data = np.array([[1,2,3],[4,5,6],[7,8,9]])

data = data.reshape(3,3)

print(data)