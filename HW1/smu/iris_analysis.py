from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris  = load_iris()
X = iris.data
y = iris.target
kfold = KFold(n_splits=5)
kf = kfold.split(X, y)


for train_index , test_index in kf:
    classifier = SVC()

    X_train , X_test = X[train_index], X[test_index]
    y_train , y_test = y[train_index], y[test_index]
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    sc = classifier.score(X, y)
    print(" full ", sc)
    print(" test ", accuracy_score(y_test, y_predict))

plt.scatter(X.transpose()[0], y)
plt.show()