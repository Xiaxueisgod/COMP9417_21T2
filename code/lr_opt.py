import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

X_train = np.load("X_train2.npy",)
X_test = np.load("X_test2.npy",)
Y_train = np.load("Y_train2.npy",)
Y_test = np.load("Y_test2.npy",)


# c_list_score = []
# for i in range(1,90):
#     model = LogisticRegression(C=i)
#     model.fit(X_train, Y_train)
#     y_pre = model.predict(X_test)
#     t = model.score(X_test,Y_test)
#     print(t)
#     c_list_score.append(t)
# print(c_list_score)
# max_score = max(c_list_score)
# print(max_score)
# max_score_c = c_list_score.index(max_score)+1
# print(f'max c is {max_score_c}')


# model_ = LogisticRegression(C=12, penalty='l1',solver='liblinear')
# model_.fit(X_train,Y_train)
# y_pre = model_.predict(X_test)
# print(classification_report(Y_test,y_pre))
# print(confusion_matrix(Y_test,y_pre))
# print(roc_auc_score(Y_test,y_pre))

