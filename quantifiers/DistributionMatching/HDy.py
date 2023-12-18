import sys

sys.path.insert(1, 'C:\\Users\\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers')

from interface_class.Quantifier import Quantifier
from utils import Quantifier_Utils as utils

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy as np
import pandas as pd

class HDy(Quantifier):

    # bins receber uma lista!
    def __init__(self, classifier, data_split='holdout'):
        self.classifier = classifier
        self.similarity_measure = 'hellinger'
        #self.bins = bins
        self.data_split = data_split
        self.p_scores = []
        self.n_scores = []
        self.test_scores = []

    def get_class_proportion(self):
        # Bins used 10 to 110 with step 11
        bin_size = np.linspace(10, 110, 11)
        # Alpha values used, 0 to 1 with step 101
        alpha_values = [round(x, 2) for x in np.linspace(0, 1, 101)]

        result = []
        for bins in bin_size:
            p_bin_count = utils.getHist(self.p_scores, bins)
            n_bin_count = utils.getHist(self.n_scores, bins)
            te_bin_count = utils.getHist(self.test_scores, bins)

            distance_list = []

            for x in range(0, len(alpha_values), 1):
                # Distance from the positives bins and the negative ones from the test
                distance = utils.Distances((p_bin_count*alpha_values[x]) + (n_bin_count*(1-alpha_values[x])), te_bin_count)
                distance = distance.hellinger()

                distance_list.append(distance)

            result.append(alpha_values[np.argmin(distance_list)])

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
        scores = scores[:,1]

        self.test_scores = scores

        return self.get_class_proportion()

if __name__ == '__main__':
    data = datasets.load_breast_cancer()

    dts_data = pd.DataFrame(data['data'], columns=data.feature_names)
    dts_data['class'] = data.target

    X = dts_data.drop(['class'], axis='columns')
    y = dts_data['class']

    X_trainn, X_testt, y_trainn, y_testt = train_test_split(X, y, test_size=0.33)

    knn = KNeighborsClassifier(n_neighbors=7)
    hdy = HDy(knn)

    hdy.fit(X_trainn, y_trainn)
    class_distribution = hdy.predict(X_testt)

    print(hdy.classifier.classes_)
    print(data.target_names)
    print(class_distribution)