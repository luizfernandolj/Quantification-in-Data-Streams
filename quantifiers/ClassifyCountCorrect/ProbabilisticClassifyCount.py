import sys

sys.path.insert(1, 'C:\\Users\\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers')

from interface_class.Quantifier import Quantifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

import pandas as pd

class ProbabilisticClassifyCount(Quantifier):

    def __init__(self, classifier):
        self.classifier = classifier

    @staticmethod
    def get_class_proportion(scores):
        total_instances = len(scores)
        number_classes = len(scores[0])

        result = [0] * number_classes

        for score in scores:
            for i in range(number_classes):
                result[i] += score[i]

        result = [round(res/total_instances, 2) for res in result]

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

    knn = KNeighborsClassifier(n_neighbors=7)
    X_trainn, X_testt, y_trainn, y_test = train_test_split(X, y, test_size=0.5)

    pcc = ProbabilisticClassifyCount(knn)
    pcc.fit(X_trainn, y_trainn)
    class_distribution = pcc.predict(X_testt)

    print(pcc.classifier.classes_)
    print(data.target_names)
    print(class_distribution)
