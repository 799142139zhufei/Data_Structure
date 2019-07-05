import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression

train = pd.read_table('zhengqi_train')
featrue_train = train.drop('target',axis = 1)
lable_train = train['target']
test_size = 0.33
random_state = 7
x_train,x_test,y_train,y_test = train_test_split(featrue_train.values,lable_train.values,
                                                 test_size=test_size,
                                                 random_state=random_state)

LR = LogisticRegression(C= 0.1,penalty='l1')
LR.fit(x_train,y_train)
test_predict = LR.predict(x_test)
recall_score = recall_score(y_test,test_predict)
print(recall_score)