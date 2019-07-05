from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

algorithms = [[GradientBoostingClassifier(n_estimators=50, random_state=1, max_depth=3),
               ['Pclass', 'Sex', 'Fare', 'Title']
               ],
              [LogisticRegression(random_state=1),
               ['Pclass', 'Sex', 'Fare', 'Title']
               ]
              ]

for a,b in algorithms:
    print(a)
    print(b)