import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

class Distances(object):

    def __init__(self, P, Q):
        if sum(P) < 1e-20 or sum(Q) < 1e-20:
            raise "One or both vector are zero (empty)..."
        if len(P) != len(Q):
            raise "Arrays need to be of equal sizes..."
        # use numpy arrays for efficient coding
        P = np.array(P, dtype=float)
        Q = np.array(Q, dtype=float)
        # Correct for zero values
        P[np.where(P < 1e-20)] = 1e-20
        Q[np.where(Q < 1e-20)] = 1e-20
        self.P = P
        self.Q = Q

    def sqEuclidean(self):
        P = self.P
        Q = self.Q
        d = len(P)
        return sum((P - Q) ** 2)

    def probsymm(self):
        P = self.P
        Q = self.Q
        d = len(P)
        return 2 * sum((P - Q) ** 2 / (P + Q))

    def topsoe(self):
        P = self.P
        Q = self.Q
        return sum(P * np.log(2 * P / (P + Q)) + Q * np.log(2 * Q / (P + Q)))

    def hellinger(self):
        P = self.P
        Q = self.Q
        return 2 * np.sqrt(1 - sum(np.sqrt(P * Q)))


def DyS_distance(sc_1, sc_2, measure):
    dist = Distances(sc_1, sc_2)

    if measure == 'topsoe':
        return dist.topsoe()
    if measure == 'probsymm':
        return dist.probsymm()
    if measure == 'hellinger':
        return dist.hellinger()
    return 100


def TernarySearch(left, right, f, eps=1e-4):
    while True:
        if abs(left - right) < eps:
            return (left + right) / 2

        leftThird = left + (right - left) / 3
        rightThird = right - (right - left) / 3

        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird


def getHist(scores, nbins):
    breaks = np.linspace(0, 1, int(nbins) + 1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks, 1.1)

    re = np.repeat(1 / (len(breaks) - 1), (len(breaks) - 1))
    for i in range(1, len(breaks)):
        re[i - 1] = (re[i - 1] + len(np.where((scores >= breaks[i - 1]) & (scores < breaks[i]))[0])) / (len(scores) + 1)
    return re


def MoSS(n, alpha, m):
    p_scores = np.random.uniform(0, 1, int(n * alpha)) ** m
    n_scores = 1 - np.random.uniform(0, 1, int(n * (1 - alpha))) ** m
    scores = pd.concat([pd.DataFrame(np.append(p_scores, n_scores)),
                        pd.DataFrame(np.append(['1'] * len(p_scores), ['2'] * len(n_scores)))], axis=1)
    scores.columns = ['score', 'label']
    return p_scores, n_scores, scores

def TPRandFPR(validation_scores):
    unique_scores = np.arange(0, 1, 0.01)
    arrayOfTPRandFPRByTr = pd.DataFrame(columns=['threshold', 'fpr', 'tpr'])
    total_positive = len(validation_scores[validation_scores['class'] == 1])
    total_negative = len(validation_scores[validation_scores['class'] == 0])

    for threshold in unique_scores:
        fp = len(validation_scores[(validation_scores['score'] > threshold) & (validation_scores['class'] == 0)])
        tp = len(validation_scores[(validation_scores['score'] > threshold) & (validation_scores['class'] == 1)])
        tpr = round(tp / total_positive, 2)
        fpr = round(fp / total_negative, 2)

        aux = pd.DataFrame([[round(threshold, 2), fpr, tpr]])
        aux.columns = ['threshold', 'fpr', 'tpr']
        arrayOfTPRandFPRByTr = pd.concat([arrayOfTPRandFPRByTr, aux])

    arrayOfTPRandFPRByTr = arrayOfTPRandFPRByTr.reset_index(drop=True)

    return arrayOfTPRandFPRByTr


def getScores(dt, label, folds):
    skf = StratifiedKFold(n_splits=folds)
    clf = RandomForestClassifier(n_estimators=200)  # put here the classifier algorithm that will be used as scorer
    results = []
    class_labl = []

    for fold_i, (train_index, valid_index) in enumerate(skf.split(dt, label)):
        tr_data = pd.DataFrame(dt.iloc[train_index])  # Train dataset and label
        tr_lbl = label.iloc[train_index]

        valid_data = pd.DataFrame(dt.iloc[valid_index])  # Validation dataset and label
        valid_lbl = label.iloc[valid_index]

        clf.fit(tr_data, tr_lbl)

        results.extend(clf.predict_proba(valid_data)[:, 1])  # evaluating scores
        class_labl.extend(valid_lbl)

    scr = pd.DataFrame(results, columns=["score"])
    scr_labl = pd.DataFrame(class_labl, columns=["class"])
    scores = pd.concat([scr, scr_labl], axis=1, ignore_index=False)

    return scores
