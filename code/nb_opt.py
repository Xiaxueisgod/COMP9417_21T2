from sklearn.model_selection import GridSearchCV,KFold
import numpy as np
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB

X_train = np.load("X_train2.npy",)
X_test = np.load("X_test2.npy",)
Y_train = np.load("Y_train2.npy",)
Y_test = np.load("Y_test2.npy",)

param_grid ={}

param_grid['alpha'] = [0.001,0.002,0.005,0.01,0.1,1.5,2,3,5]

#model = BernoulliNB()
model = MultinomialNB()

kfold = KFold(n_splits=10)

grid = GridSearchCV(estimator= model, param_grid = param_grid, cv=kfold)

grid_result = grid.fit(X = X_train, y = Y_train)

print('The best score：%s The best classifier%s'%(grid_result.best_score_,grid_result.best_params_))
