import sys

sys.path.insert(1, 'C:\\Users\\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers')

from interface_class.Quantifier import Quantifier
from utils.Quantifier_Utils import TPRandFPR

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

import pandas as pd

class T50(Quantifier):

    def __init__(self, classifier):
        self.classifier = classifier
        self.tprfpr = None

    def get_class_proportion(self, scores):
        # Taking threshold tpr and fpr where tpr=50% is minimum
        index = abs(self.tprfpr['tpr'] - 0.5).idxmin()

        threshold, fpr, tpr = self.tprfpr.loc[index]

        # Getting the positive scores proportion
        pos_scores = [score[1] for score in scores]
        class_prop = len([pos_score for pos_score in pos_scores if pos_score >= threshold])
        class_prop /= len(scores)

        diff_tpr_fpr = abs(tpr - fpr)

        if diff_tpr_fpr == 0:
            pos_prop = round(abs(class_prop))
        else:
            pos_prop = round(abs(class_prop - fpr) / abs(tpr - fpr), 2)  # adjusted class proportion

        if pos_prop <= 0:  # clipping the output between [0,1]
            pos_prop = 0
        elif pos_prop >= 1:
            pos_prop = 1

        neg_prop = round(1 - pos_prop, 2)

        return [neg_prop, pos_prop]


    def fit(self, X_train, y_train):
        # Validation split
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train)

        # Validation train
        self.classifier.fit(X_val_train, y_val_train)

        # Validation result_table
        pos_val_scores = self.classifier.predict_proba(X_val_test)[:, 1]

        # Generating the dataframe with the positive scores and it's own class
        pos_val_scores = pd.DataFrame(pos_val_scores, columns=['score'])
        pos_val_labels = pd.DataFrame(y_val_test, columns=['class'])

        # Needed to reset the index, predict_proba result_table reset the indexes!
        pos_val_labels.reset_index(drop=True, inplace=True)

        val_scores = pd.concat([pos_val_scores, pos_val_labels], axis='columns', ignore_index=False)

        # Generating the tpr and fpr for thresholds between [0, 1] for the validation scores!
        self.tprfpr = TPRandFPR(val_scores)

        # Fit the classifier again but now with the whole train set
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        # Result from the test
        scores = self.classifier.predict_proba(X_test)

        # Class proportion generated through the main algorithm
        return self.get_class_proportion(scores)


if __name__ == '__main__':
    data = datasets.load_breast_cancer()

    dts_data = pd.DataFrame(data['data'], columns=data.feature_names)
    dts_data['class'] = data.target

    X = dts_data.drop(['class'], axis='columns')
    y = dts_data['class']

    X_trainn, X_testt, y_trainn, y_testt = train_test_split(X, y, test_size=0.33)

    knn = KNeighborsClassifier(n_neighbors=7)
    t50_qtf = T50(knn)

    t50_qtf.fit(X_trainn, y_trainn)
    class_distribution = t50_qtf.predict(X_testt)

    print(t50_qtf.classifier.classes_)
    print(data.target_names)
    print(class_distribution)