import  tushare as ts

df = ts.profit_data(year= 2018,top= 10)
print(df[df['divi']>2])