import sys

sys.path.insert(1, 'C:\\Users\\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers')

from interface_class.Quantifier import Quantifier
from utils import Quantifier_Utils as utils

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

import pandas as pd
import numpy as np

class DyS(Quantifier):

    # bins receber uma lista!
    def __init__(self, classifier, similarity_measure, data_split='holdout'):
        self.classifier = classifier
        self.similarity_measure = similarity_measure
        #self.bins = bins
        self.data_split = data_split
        self.p_scores = []
        self.n_scores = []
        self.test_scores = []

    def get_class_proportion(self):
        # Bins used 2 to 20 with step 10, and 30 in the end
        bin_size = np.linspace(2, 20, 10)
        bin_size = np.append(bin_size, 30)

        result = []
        for bins in bin_size:
            p_bin_count = utils.getHist(self.p_scores, bins)
            n_bin_count = utils.getHist(self.n_scores, bins)
            te_bin_count = utils.getHist(self.test_scores, bins)

            def f(x):
                return utils.DyS_distance(((p_bin_count * x) + (n_bin_count * (1 - x))), te_bin_count, measure=self.similarity_measure)

            result.append(utils.TernarySearch(0, 1, f))

        pos_prop = round(float(np.median(result)), 2)
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
        scores = [score[1] for score in scores]

        self.test_scores = scores

        return self.get_class_proportion()

if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    dts_data = pd.DataFrame(data['data'], columns=data.feature_names)
    dts_data['class'] = data.target

    X = dts_data.drop(['class'], axis='columns')
    y = dts_data['class']

    knn = KNeighborsClassifier(n_neighbors=7)
    X_trainn, X_testt, y_trainn, y_test = train_test_split(X, y, test_size=0.5)

    sord = DyS(knn, 'topsoe')
    sord.fit(X_trainn, y_trainn)
    class_distribution = sord.predict(X_testt)

    print(sord.classifier.classes_)
    print(data.target_names)
    print(class_distribution)