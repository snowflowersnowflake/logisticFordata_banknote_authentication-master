# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression

# 读取数据
train = pd.read_csv("data_banknote_authentication_TrainData.csv")
test = pd.read_csv("data_banknote_authentication_TestingData .csv")
submit = pd.read_csv("sample_submit.csv")
# 删除id
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# 取出训练集的y
y_train = train.pop('Evaluation')

# 建立LASSO逻辑回归模型
clf = LogisticRegression(penalty='l1', C=1.0, random_state=0,solver='liblinear')
clf.fit(train, y_train)
y_pred = clf.predict_proba(test)[:, 1]

# 输出预测结果至my_LASSO_prediction.csv
submit['Evaluation'] = y_pred
submit.to_csv('my_LASSO_prediction.csv', index=False)