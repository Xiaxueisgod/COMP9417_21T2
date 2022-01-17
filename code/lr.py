import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

X_train = np.load("X_train.npy",)
X_test = np.load("X_test.npy",)
Y_train = np.load("Y_train.npy",)
Y_test = np.load("Y_test.npy",)

model = LogisticRegression()
model.fit(X_train,Y_train)
y_pre = model.predict(X_test)
print(classification_report(Y_test,y_pre))
print(confusion_matrix(Y_test,y_pre))
print(roc_auc_score(Y_test,y_pre))
