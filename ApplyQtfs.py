import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from quantifiers.ClassifyCountCorrect.AdjustedClassifyCount import AdjustedClassifyCount
from quantifiers.ClassifyCountCorrect.ClassifyCount import ClassifyCount
from quantifiers.ClassifyCountCorrect.MAX import MAX
from quantifiers.ClassifyCountCorrect.MedianSweep import MedianSweep
from quantifiers.ClassifyCountCorrect.ProbabilisticAdjustedClassifyCount import ProbabilisticAdjustedClassifyCount
from quantifiers.ClassifyCountCorrect.ProbabilisticClassifyCount import ProbabilisticClassifyCount
from quantifiers.ClassifyCountCorrect.T50 import T50
from quantifiers.ClassifyCountCorrect.X import Xqtf
from quantifiers.DistributionMatching.DyS import DyS
from quantifiers.DistributionMatching.HDy import HDy
from quantifiers.DistributionMatching.SORD import SORD

class ApplyQtfs:
  def __init__(self, trainX, trainy, window, model, thr):
    self.trainX = trainX
    self.trainy = trainy
    self.window = window
    self.model = model
    self.thr = thr
    self.quantifiers = ["CC", "ACC", "MS", "DyS"]
    self.quantifiers_initialized = {}

  def apply_quantifier(self, quantifier, clf, thr, measure, train, test):
    if quantifier not in self.quantifiers_initialized:
      if quantifier == "CC":
        cc = ClassifyCount(classifier=clf, threshold=thr)
        cc.fit(train[0], train[1])
        self.quantifiers_initialized["CC"] = cc

        return cc.predict(test)
      if quantifier == "ACC":
        acc = AdjustedClassifyCount(classifier=clf, threshold=thr)
        acc.fit(train[0], train[1])
        self.quantifiers_initialized["ACC"] = acc

        return acc.predict(test)
      if quantifier == "PCC":
        pcc = ProbabilisticClassifyCount(classifier=clf)
        pcc.fit(train[0], train[1])
        self.quantifiers_initialized["PCC"] = pcc

        return pcc.predict(test)

      if quantifier == "PACC":
        pacc = ProbabilisticAdjustedClassifyCount(classifier=clf, threshold=thr)
        pacc.fit(train[0], train[1])
        self.quantifiers_initialized["PACC"] = pacc

        return pacc.predict(test)

      if quantifier == "X":
        x_qtf = Xqtf(classifier=clf)
        x_qtf.fit(train[0], train[1])
        self.quantifiers_initialized["X"] = x_qtf

        return x_qtf.predict(test)

      if quantifier == "MAX":
        max_qtf = MAX(classifier=clf)
        max_qtf.fit(train[0], train[1])
        self.quantifiers_initialized["MAX"] = max_qtf

        return max_qtf.predict(test)

      if quantifier == "T50":
        t50 = T50(classifier=clf)
        t50.fit(train[0], train[1])
        self.quantifiers_initialized["T50"] = t50

        return t50.predict(test)

      if quantifier == "MS":
        ms = MedianSweep(classifier=clf)
        ms.fit(train[0], train[1])
        self.quantifiers_initialized["MS"] = ms

        return ms.predict(test)

      if quantifier == "HDy":
        hdy = HDy(classifier=clf)
        hdy.fit(train[0], train[1])
        self.quantifiers_initialized["HDy"] = hdy

        return hdy.predict(test)

      if quantifier == "DyS":
        dys = DyS(classifier=clf, similarity_measure=measure)
        dys.fit(train[0], train[1])
        self.quantifiers_initialized["DyS"] = dys

        return dys.predict(test)

      if quantifier == "SORD":
        sord = SORD(classifier=clf)
        sord.fit(train[0], train[1])
        self.quantifiers_initialized["SORD"] = sord

        return sord.predict(test)
    else:
      return self.quantifiers_initialized[quantifier].predict(test)

  def aplly_qtf(self):
    proportions = {}
    for qtf in self.quantifiers:
      # .............Calling of Methods.................
        pred_pos_prop = self.apply_quantifier(quantifier=qtf, clf=self.model,
                                            thr=self.thr,
                                            measure='topsoe',
                                            train=[self.trainX, self.trainy],
                                            test=self.window)
        pred_pos_prop = [round(pred_pos_prop[0], 2), round(1 - pred_pos_prop[0], 2)]# Getting the proportions
        print(pred_pos_prop)
        proportions[qtf] = pred_pos_prop
    return proportions
      
                           