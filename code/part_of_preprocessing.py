# encoding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

warnings.filterwarnings('ignore')
from scipy.stats import skew

train_ = pd.read_csv('act_train.csv')
test_ = pd.read_csv('act_test.csv')
train = pd.DataFrame(train_).iloc[0:10000, :]
test = pd.DataFrame(test_).iloc[0:10000,:]
print(train)
people = pd.read_csv('people.csv')

activity_id = test['activity_id']

# 将人的id转换为int类型
people['people_id'] = people['people_id'].apply(lambda x: x.split('_')[1])
people['people_id'] = pd.to_numeric(people['people_id']).astype(int)

train['people_id'] = train['people_id'].apply(lambda x: x.split('_')[1])
train['people_id'] = pd.to_numeric(train['people_id']).astype(int)

test['people_id'] = test['people_id'].apply(lambda x: x.split('_')[1])
test['people_id'] = pd.to_numeric(test['people_id']).astype(int)

train = train.drop('activity_id', axis=1)
train = train.drop('date', axis=1)
test = test.drop('activity_id', axis=1)
test = test.drop('date', axis=1)

string_feature = train.select_dtypes(include=['object'])
string_feature_test = test.select_dtypes(include=['object'])

# 对训练集特征转换为数值型
for i in string_feature.columns:
    string_feature[i] = string_feature[i].fillna("type 0")  # 空值使用0填充
    string_feature[i] = string_feature[i].apply(lambda x: x.split(" ")[1])  # 转换为数据
    string_feature[i] = pd.to_numeric(string_feature[i])
# 对测试集特征转换为数值型
for i in string_feature.columns:
    string_feature_test[i] = string_feature_test[i].fillna("type 0")
    string_feature_test[i] = string_feature_test[i].apply(lambda x: x.split(" ")[1])
    string_feature_test[i] = pd.to_numeric(string_feature_test[i])

train_new = string_feature
train_new['people_id'] = train['people_id']
y = train['outcome']

test_new = string_feature_test
test_new['people_id'] = test['people_id']

people = people.drop('date', axis=1)
string_feature_people = people.select_dtypes(include=['object'])
bool_feature_people = people.select_dtypes(include=['bool'])

# 对people.csv中的string类型的数据转换为数值型
for i in string_feature_people.columns:
    string_feature_people[i] = string_feature_people[i].fillna("type 0")
    string_feature_people[i] = string_feature_people[i].apply(lambda x: x.split(" ")[1])
    string_feature_people[i] = pd.to_numeric(string_feature_people[i]).astype(int)

# 将布尔型特征转换为0 1形式
from sklearn.preprocessing import LabelEncoder

for i in bool_feature_people.columns:
    lb = LabelEncoder()
    lb.fit(list(bool_feature_people[i].values))
    bool_feature_people[i] = lb.transform(list(bool_feature_people[i].values))

people_new = (pd.concat([string_feature_people, bool_feature_people], axis=1))
people_new['people_id'] = people['people_id']
people_new['char_38'] = people['char_38']

# 合并people.csv的特征和train中的特征
total_train = train_new.merge(people_new, on='people_id', how="left")
total_test = test_new.merge(people_new, on='people_id', how="left")

# 数据集划分
X_train, X_test, Y_train, Y_test = train_test_split(total_train, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

np.save("X_train2", X_train)
np.save("X_test2", X_test)
np.save("Y_train2", Y_train)
np.save("Y_test2", Y_test)
print("ok")
