from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class GridSearch:
    def run(self, classifier , data , parameters):
        ret ={}
        cls = classifier(**parameters)
        X, y = data
        fold = KFold(n_splits=2)
        for idx, (train_idx,  test_idx)  in enumerate(fold.split(X,y)):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            cls.fit(X_train, y_train)
            pred = cls.predict(X_test)
            acc = accuracy_score(y_test, pred)
            ret[idx] = {'clf' : cls,
                        'train_idx' : train_idx,
                        'test_idx' : test_idx,
                        'accuracy': acc}

        return ret
