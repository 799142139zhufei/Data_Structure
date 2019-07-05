from sklearn.metrics import confusion_matrix
import pandas as pd

test = ['1','0','0','1','1','1']
predict = ['1','0','1','0','1','0']

cnf_matrix = confusion_matrix(test,predict)
df = pd.DataFrame(cnf_matrix)
print(df)
print(cnf_matrix[0,1])
