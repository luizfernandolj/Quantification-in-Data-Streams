import sys

sys.path.insert(1, 'C:\\Users\\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers')

from interface_class.Quantifier import Quantifier
from utils.Quantifier_Utils import TPRandFPR

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

import pandas as pd

class AdjustedClassifyCount(Quantifier):

    def __init__(self, classifier, threshold=0.5):
        self.classifier = classifier
        self.threshold = threshold
        self.tprfpr = None

    def get_class_proportion(self, scores):
        total_instances = len(scores)
        number_classes = len(scores[0])
        result = [0] * number_classes

        # Counting positive and negative classes
        for score in scores:
            if score[1] >= self.threshold:
                result[1] += 1
            else:
                result[0] += 1

        # Choosing the tpr and fpr based on what was passed in the constructor
        self.tprfpr = self.tprfpr[self.tprfpr['threshold'] == self.threshold]
        diff_tprfpr = self.tprfpr['tpr'] - self.tprfpr['fpr']

        pos_classify_proportion = result[1] / total_instances
        pos_classify_proportion = round(pos_classify_proportion, 2)

        # Counting result_table for positive class
        if diff_tprfpr.iloc[0] == 0:
            pos_adjusted = pos_classify_proportion
        else:
            pos_adjusted = (pos_classify_proportion - self.tprfpr['fpr']) / diff_tprfpr
            # Transforming to float
            pos_adjusted = float(pos_adjusted.iloc[0])

        # Ensuring the result_table between 0 and 1
        if pos_adjusted <= 0:
            pos_adjusted = 0
        elif pos_adjusted >= 1:
            pos_adjusted = 1

        neg_adjusted = round(1 - pos_adjusted, 2)
        result = [neg_adjusted, round(pos_adjusted, 2)]

        return result


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
        # Result from the train
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
    acc = AdjustedClassifyCount(knn, 0.5)
    acc.fit(X_trainn, y_trainn)
    class_distribution = acc.predict(X_testt)

    print(acc.classifier.classes_)
    print(data.target_names)
    print(class_distribution)