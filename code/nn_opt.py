import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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

mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(100,), (100, 30)],
                             "solver": ['adam', 'sgd', 'lbfgs'],
                             "max_iter": [20],
                             "verbose": [True]
                             }

model = MLPClassifier(max_iter=400)
#model = GridSearchCV(mlp,mlp_clf__tuned_parameters,n_jobs=2)
model.fit(X_train,Y_train)

# model.fit(X_train,Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], eval_metric='logloss', verbose=True)
# y_pre = model.predict(X_test)
# print(classification_report(Y_test,y_pre))
# print(confusion_matrix(Y_test,y_pre))
# print("roc result = {:.3f}".format(roc_auc_score(Y_test,y_pre)))
y_pre = model.predict(X_test)
print(classification_report(Y_test,y_pre))
print(confusion_matrix(Y_test,y_pre))
print(roc_auc_score(Y_test,y_pre))
plot_roc(Y_test,y_pre)