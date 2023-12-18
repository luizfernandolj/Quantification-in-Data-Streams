import sys

sys.path.insert(1, 'C:\\Users\\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers')
from interface_class.Quantifier import Quantifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

import pandas as pd
import numpy as np

class ClassifyCount(Quantifier):

    def __init__(self, classifier, threshold=0.5):
        self.classifier = classifier
        self.threshold = threshold

    def get_class_proportion(self, scores):
        total_instances = len(scores)
        number_classes = len(scores[0])
        result = [0] * number_classes

        if number_classes > 2:
            for score in scores:
                target = np.argmax(score)
                result[target] += 1
        else:
            for score in scores:
                if score[0] >= self.threshold:
                    result[0] += 1
                else:
                    result[1] += 1

        result = [round(proportion/total_instances, 2) for proportion in result]

        return result

    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        scores = self.classifier.predict_proba(X_test)
        return self.get_class_proportion(scores)

if __name__ == '__main__':
    data = datasets.load_breast_cancer()

    dts_data = pd.DataFrame(data['data'], columns=data.feature_names)
    dts_data['class'] = data.target

    X = dts_data.drop(['class'], axis='columns')
    y = dts_data['class']

    X_trainn, X_testt, y_trainn, y_testt = train_test_split(X, y, test_size=0.33)

    knn = KNeighborsClassifier(n_neighbors=7)
    hdy = ClassifyCount(knn)

    hdy.fit(X_trainn, y_trainn)
    class_distribution = hdy.predict(X_testt)

    print(hdy.classifier.classes_)
    print(data.target_names)
    print(class_distribution)