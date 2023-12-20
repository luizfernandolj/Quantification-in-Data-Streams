import sys

sys.path.insert(1, 'C:\\Users\\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers')

from interface_class.Quantifier import Quantifier
from utils.Quantifier_Utils import TPRandFPR

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

import pandas as pd
import numpy as np

class MedianSweep(Quantifier):

    def __init__(self, classifier):
        self.classifier = classifier
        self.tprfpr = None

    def get_class_proportion(self, scores):
        # Threshold values from 0.01 to 0.99
        unique_scores = np.arange(0.01, 1, 0.01)
        prevalances_array = []

        for i in unique_scores:
            threshold = round(i, 2)

            # List of tpr and fpr by threshold
            record = self.tprfpr[self.tprfpr['threshold'] == threshold]

            # Getting tpr and fpr from determined threshold
            tpr = record['tpr']
            fpr = record['fpr']

            batch_size = len(scores)

            pos_scores = [score[1] for score in scores]

            estimated_positive_ratio = len([pos_score for pos_score in pos_scores if pos_score >= threshold])
            estimated_positive_ratio /= batch_size

            diff_tpr_fpr = abs(tpr - fpr)
            
            if diff_tpr_fpr.tolist()[0] == 0:
                final_prevalence = round(abs(estimated_positive_ratio))
            else:
                # Calculating the positive class proportion
                final_prevalence = round(abs(estimated_positive_ratio - fpr) / diff_tpr_fpr, 2).tolist()[0]

            # Appending to the array
            prevalances_array.append(final_prevalence)

        # Picking median result
        pos_prop = float(np.median(prevalances_array))

        pos_prop = round(pos_prop, 2)

        # Clipping the output between [0,1]
        if pos_prop <= 0:
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

        # Generating the dataframe with the positive scores and its own class
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

    ms = MedianSweep(knn)

    ms.fit(X_trainn, y_trainn)
    class_distribution = ms.predict(X_testt)

    print(ms.classifier.classes_)
    print(class_distribution)