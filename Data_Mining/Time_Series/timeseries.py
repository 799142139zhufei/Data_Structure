#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pandas as pd


pd = pd.date_range('2018/01/01',periods=10,freq='M')
print(pd)