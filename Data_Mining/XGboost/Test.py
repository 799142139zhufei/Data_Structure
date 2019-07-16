import pandas as pd

results_table = pd.DataFrame([[0.01,10],[0.1,100],[1,1000],[10,10000]],
                             columns=['a','b'])
print(results_table[0:1]['a'])




