from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

class ML2GridSearch:

    def run(self, classifier , data , parameters):
        ret ={}
        cls = classifier(**parameters)
        X, y = data
        fold = KFold(n_splits=5)

        for idx, (train_idx,  test_idx)  in enumerate(fold.split(X,y)):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            cls.fit(X_train, y_train)
            pred = cls.predict(X_test)
            acc = accuracy_score(y_test, pred)
            ret[type(cls).__name__+  "_"+str(idx)] = \
                        {'clf' : cls,
                        'train_idx' : train_idx,
                        'test_idx' : test_idx,
                        'accuracy': acc,
                        'params':parameters
                         }
        return ret

    def run_with_mul_params(self,  data , parameters):
        result ={}
        for cls , param in parameters.items():
            res = grid_search.run(cls, data, param)
            result.update(res)
        # sort result with decreasing accuracy
        result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1]['accuracy'], reverse= True)}
        return result


def prepare_data():
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    data = (X, y)
    return data


def get_classifier_param_set():
    classifier_1 = Perceptron
    param_perceptron = {'tol' : 1e-3, 'random_state' : 0}
    classifier_2 = SVC
    param_1_SVC = {'C': 2, 'class_weight': 'balanced'}
    classifier_3 = GaussianNB
    params_gauss_nb =  {}
    classifier_4 = SGDClassifier
    param_sgd=  {'max_iter': 1000, 'tol': 1e-3}
    param_set = {classifier_1: param_perceptron, classifier_2:param_1_SVC , classifier_3:params_gauss_nb,
                 classifier_4:param_sgd}
    return param_set


def plot_result(result):
    acc_scores = [result[res]['accuracy'] for res in result]
    clf_names = [res for res in result]
    plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.bar(clf_names, acc_scores)
    plt.xlabel('classifier')
    plt.ylabel('accuracy score')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=.3)
    # plt.show()
    plt.savefig('grid_search_output.png')
    plt.show()

if __name__ == "__main__":
    data = prepare_data()
    param_set = get_classifier_param_set()
    grid_search = ML2GridSearch()
    result = grid_search.run_with_mul_params(data, param_set)
    plot_result(result)