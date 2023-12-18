import sys

sys.path.insert(1, 'C:\\Users\\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers')

from interface_class.Quantifier import Quantifier

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

class SORD(Quantifier):

    def __init__(self, classifier, data_split='holdout'):
        self.classifier = classifier
        self.data_split = data_split
        self.p_scores = []
        self.n_scores = []
        self.test_scores = []

    def get_class_proportion(self):

        alpha = np.linspace(0, 1, 101)
        sc_1 = self.p_scores
        sc_2 = self.n_scores
        ts = self.test_scores

        vDist = []

        for k in alpha:
            pos = np.array(sc_1)
            neg = np.array(sc_2)
            test = np.array(ts)
            pos_prop = k

            p_w = pos_prop / len(pos)
            n_w = (1 - pos_prop) / len(neg)
            t_w = -1 / len(test)

            p = list(map(lambda x: (x, p_w), pos))
            n = list(map(lambda x: (x, n_w), neg))
            t = list(map(lambda x: (x, t_w), test))

            v = sorted(p + n + t, key=lambda x: x[0])

            acc = v[0][1]
            total_cost = 0

            for i in range(1, len(v)):
                cost_mul = v[i][0] - v[i - 1][0]
                total_cost = total_cost + abs(cost_mul * acc)
                acc = acc + v[i][1]

            vDist.append(total_cost)

        pos_prop = round(alpha[vDist.index(min(vDist))], 2)
        neg_prop = round(1 - pos_prop, 2)

        return [neg_prop, pos_prop]

    def fit(self, X_train, y_train):
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, test_size=0.33)

        self.classifier.fit(X_val_train, y_val_train)
        scores = self.classifier.predict_proba(X_val_test)

        p_scores = scores[y_val_test == 1]
        self.p_scores = [p_score[1] for p_score in p_scores]

        n_scores = scores[y_val_test == 0]
        self.n_scores = [n_score[1] for n_score in n_scores]

        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        scores = self.classifier.predict_proba(X_test)
        scores = scores[:,1]

        self.test_scores = scores

        return self.get_class_proportion()


if __name__ == '__main__':
    dts_data = pd.read_csv('C:\\1Faculdade\\EstagioSupervisionado\\AedesQuinx.csv')
    dts_data.rename(columns={'species': 'class'}, inplace=True)

    mapping = {'AA': 1, 'CQ': 0}
    dts_data['class'] = dts_data['class'].map(mapping)

    size = 500
    pos_class = dts_data[dts_data['class'] == 1]
    neg_class = dts_data[dts_data['class'] == 0]

    pos_class = pos_class.sample(n=size)
    neg_class = neg_class.sample(n=size)

    dts_data = pd.concat([pos_class, neg_class], ignore_index=True)

    print(dts_data)


    label = dts_data['class']
    dataset = dts_data.drop(['class'], axis='columns')

    knn = KNeighborsClassifier(n_neighbors=7)
    X_trainn, X_testt, y_trainn, y_test = train_test_split(dataset, label, test_size=0.5)

    sord = SORD(knn)
    sord.fit(X_trainn, y_trainn)
    class_distribution = sord.predict(X_testt)

    print(sord.classifier.classes_)
   # print(data.target_names)
    print(class_distribution)
