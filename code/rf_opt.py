import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_roc(labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

X_train = np.load("X_train.npy",)
X_test = np.load("X_test.npy",)
Y_train = np.load("Y_train.npy",)
Y_test = np.load("Y_test.npy",)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train,Y_train)
y_pre = model.predict(X_test)
print(classification_report(Y_test,y_pre))
print(confusion_matrix(Y_test,y_pre))
print(roc_auc_score(Y_test,y_pre))
plot_roc(Y_test,y_pre)