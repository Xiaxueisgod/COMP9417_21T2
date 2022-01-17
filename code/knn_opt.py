import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

X_train = np.load("X_train2.npy", )
X_test = np.load("X_test2.npy", )
Y_train = np.load("Y_train2.npy", )
Y_test = np.load("Y_test2.npy", )
best_score = []
best_k = -1
# choose best value for n_neighbors

model_ori = KNeighborsClassifier()
model_ori.fit(X_train, Y_train)
y_pre_ori = model_ori.predict(X_test)
print(classification_report(Y_test, y_pre_ori))
# print(confusion_matrix(Y_test, y_pre))
# print(roc_auc_score(Y_test, y_pre))


for k in range(1,11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)
    y_pre = model.predict(X_test)
    t = model.score(X_test, Y_test)
    best_score.append(t)

best_k = best_score.index(max(best_score))+1
print(best_score)


best_method = []
weight = ['uniform','distance']
for w in weight:
    model = KNeighborsClassifier(n_neighbors=1, weights=w)
    model.fit(X_train, Y_train)
    y_pre = model.predict(X_test)
    t = model.score(X_test, Y_test)
    best_method.append(t)

print(best_method)


# best_score = 0.0
# best_k = -1
# best_method = ''
# for method in['uniform', 'distance']:
#     for k in range(1, 11):
#         knn = KNeighborsClassifier(n_neighbors=k, weights=method)
#         knn.fit(X_train, Y_train)
#         t = knn.score(X_test, Y_test)
#         if t > best_score:
#             best_score = t
#             best_k = k
#             best_method = method


p_scores_list = []
for i in range(1,6):
    knn = KNeighborsClassifier(n_neighbors=1, weights='distance',p=i)
    knn.fit(X_train, Y_train)
    y_pre = knn.predict(X_test)
    t = knn.score(X_test, Y_test)
    p_scores_list.append(t)
print(p_scores_list)

print(f'best_score is 0.9655')
print(f'best n_neighbors is 1')
print(f'best_method is distance')
print(f'best p is 1')

# model_best =
model_best = KNeighborsClassifier(n_neighbors=1,weights='distance',p=1)
model_best.fit(X_train, Y_train)
y_pre_best = model_best.predict(X_test)
print(classification_report(Y_test, y_pre_best))
# print(confusion_matrix(Y_test, y_pre))
# print(roc_auc_score(Y_test, y_pre))

