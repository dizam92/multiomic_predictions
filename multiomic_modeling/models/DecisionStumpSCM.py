# -*- coding: utf-8 -*-
__author__ = 'maoss2'

from pyscm.scm import SetCoveringMachineClassifier as scm

from sklearn.base import BaseEstimator, ClassifierMixin
from six import iteritems


class DecisionStumpSCMNew(BaseEstimator, ClassifierMixin):
    """
    A hands on class of SCM using decision stump, built with sklearn format in order to use sklearn function on SCM like
    CV, gridsearch, and so on ...
    """

    def __init__(self, model_type='conjunction', p=0.1, max_rules=10, random_state=42):
        super(DecisionStumpSCMNew, self).__init__()
        self.model_type = model_type
        self.p = p
        self.max_rules = max_rules
        self.random_state = random_state

    def fit(self, X, y):
        self.clf = scm(model_type=self.model_type, max_rules=self.max_rules, p=self.p, random_state=self.random_state)
        self.clf.fit(X=X, y=y)

    def predict(self, X):
        return self.clf.predict(X)

    def set_params(self, **params):
        for key, value in iteritems(params):
            if key == 'p':
                self.p = value
            if key == 'model_type':
                self.model_type = value
            if key == 'max_rules':
                self.max_rules = value
        return self
    
    def get_stats(self):
        return {"Binary_attributes": self.clf.model_.rules}