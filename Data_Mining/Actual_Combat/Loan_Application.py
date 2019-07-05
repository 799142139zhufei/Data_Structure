#!/usr/bin/python3
# -*- coding:utf-8 -*- #编码声明，不要忘记！

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_predict


'''
业务场景描述：贷款申请最大化利润
每个人来贷款根据其特征是否贷款/贷款的额度应该是多少
'''
#1、数据加载
'''
data = pd.read_csv('LoanStats3a.csv',skiprows=1) # 忽视第一行
data_num = len(data)/2
loan_2007 = data.dropna(thresh=data_num,axis=1) # 基于列----删除缺失值该列中data_num
loan_2007 = loan_2007.drop(['desc', 'url'],axis=1)
loan_2007.to_csv('Loan_2007.csv',index = False)
'''
loan_2007 = pd.read_csv('Text/Loan_2007.csv')

# 结合实际业务场景---删除无关的特征
loan_2007 = loan_2007.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv",
                            "grade", "sub_grade", "emp_title", "issue_d"],
                             axis=1)
loan_2007 = loan_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt",
                            "total_pymnt_inv", "total_rec_prncp"],
                             axis=1)
loan_2007 = loan_2007.drop(["total_rec_int", "total_rec_late_fee", "recoveries",
                            "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"],
                             axis=1)

#value_counts = loan_2007['loan_status'].value_counts()

#2、数据预处理

# 确定label基于Fully Paid(同意放款)、Charged Off(不同意放款)删除其它label
columns = loan_2007.columns.tolist()
drop_columns = []
for column in columns:
    column_num = loan_2007[column].dropna().unique()
    if len(column_num) ==1:
        drop_columns.append(column)
loan_2007 = loan_2007.drop(drop_columns,axis=1) # 删除该列只有一个值的数据消除噪声数据

# 将label（Fully Paid、Charged Off）进行转换
loan_2007 = loan_2007[(loan_2007['loan_status'] == 'Fully Paid')|
                      (loan_2007['loan_status'] == 'Charged Off')]
def label(loans):
    if loans == 'Fully Paid':
        return 1
    elif loans == 'Charged Off':
        return 0
loan_2007['loan_status'] = loan_2007['loan_status'].apply(lambda x : label(x))

# 删除缺失值比较多的列且存在行缺失值的也删除
loan_2007 = loan_2007.drop(['pub_rec_bankruptcies'],axis= 1)
loan_2007 = loan_2007.dropna(axis=0)
# loan_2007.dtypes.value_counts() 对特征的属性进行分组
# object_columns_df = loan_2007.select_dtypes(include= 'object')  需要将那些字符类型进行转换

# 对那些包含%的列进行变换
'''
loan_2007["int_rate"] = loan_2007["int_rate"].str.rstrip("%").astype("float")
loan_2007["revol_util"] = loan_2007["revol_util"].str.rstrip("%").astype("float")
'''
def revol(x):
   x = x.replace('%','')
   return x
loan_2007['revol_util'] = loan_2007['revol_util'].apply(lambda x : revol(x))
loan_2007['int_rate'] = loan_2007['int_rate'].apply(lambda x : revol(x))

# 对年份进行变换
mapping_dict = {
    'emp_length': {
        '10+ years': 10,
        '9 years': 9,
        '8 years': 8,
        '7 years': 7,
        '6 years': 6,
        '5 years': 5,
        '4 years': 4,
        '3 years': 3,
        '2 years': 2,
        '1 year': 1,
        '< 1 year': 0,
        'n/a': 0
    }
}

loan_2007 = loan_2007.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)
loan_2007 = loan_2007.replace(mapping_dict)

# 对特征进行one-hot编码
cat_columns = ["home_ownership", "verification_status", "emp_length", "purpose", "term"]
dummy_df = pd.get_dummies(loan_2007[cat_columns])
loan_2007 = pd.concat([loan_2007, dummy_df], axis=1)
loan_2007 = loan_2007.drop(cat_columns, axis=1)
loan_2007 = loan_2007.drop("pymnt_plan", axis=1)

#4、模型建立、预测和评估------对于正负样本不均时（向上采样、向下采样和调整正负样本的权重）
features = loan_2007.drop('loan_status',axis = 1)
target = loan_2007['loan_status']

# rf = RandomForestClassifier(n_estimators=10,class_weight="balanced", random_state=1) 随机深林==调整权重
# lr = LogisticRegression(class_weight= 'balanced') #  逻辑回归==调整权重
# 自定义权重
penalty = {
    0: 5,
    1: 1
}
lr = LogisticRegression(class_weight= penalty) #  调整权重

kf = KFold(n_splits=5,random_state=1)
predictions = cross_val_predict(lr,features.values,target,cv = kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loan_2007["loan_status"] == 0)
print(fp_filter)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loan_2007["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loan_2007["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loan_2007["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print(fpr)