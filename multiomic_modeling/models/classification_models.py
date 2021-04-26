import numpy as np 
import pandas as pd
import json
import logging
from collections import defaultdict
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from multiomic_modeling.loss_and_metrics import ClfMetrics

logging.getLogger('parso.python.diff').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nb_jobs = 32
cv_fold = KFold(n_splits=5, shuffle=True, random_state=42)
parameters_dt = {'max_depth': np.arange(1, 5),  # Moins de profondeur pour toujours eviter l'overfitting
                 'min_samples_split': np.arange(2, 15),  # Eviter les small value pour eviter l'overfitting
                 'criterion': ['gini', 'entropy']
                 }
parameters_rf = {'max_depth': np.arange(1, 5),
                 'min_samples_split': np.arange(2, 15),
                 'criterion': ['gini', 'entropy'],
                 'n_estimators': [25, 50, 75, 100]
                 }
balanced_weights = {0: 4.1472332, 1: 0.87510425, 2: 0.30869373, 3: 1.2229021 , 
                    4: 8.47878788, 5: 0.7000834, 6: 7.94886364, 7: 1.87032086, 
                    8: 0.63379644, 9: 0.63169777, 10: 4.19280719, 11: 0.40417951, 
                    12: 1.08393595, 13: 1.90772727, 14: 0.72125795, 15: 0.87110834, 
                    16: 0.59523472, 17: 0.61243251, 18: 4.38557994, 19: 0.63169777,
                    20: 1.94666048, 21: 2.04035002, 22: 0.67410858, 23: 2.08494784, 
                    24: 1.40791681, 25: 0.79654583, 26: 0.74666429, 27: 2.74493133, 
                    28: 0.65783699, 29: 3.02813853, 30: 0.65445189, 31: 6.6937799, 
                    32: 4.76931818
            }

class TreeAndForestTemplate():
    def __init__(self, algo='tree', params_dt=parameters_dt, params_rf=parameters_rf, nb_jobs=nb_jobs, cv=cv_fold):
        super(TreeAndForestTemplate, self).__init__()
        assert type(algo) == str, 'algo must be either tree or rf'
        assert type(params_dt) == dict, 'params_dt must be a dictionary'
        assert type(parameters_rf) == dict, 'parameters_rf must be a dictionary'
        self.nb_jobs = nb_jobs
        self.cv = cv
        if algo == 'tree':
            self.learner = DecisionTreeClassifier(random_state=42, class_weight=balanced_weights)
            self.params_dt = params_dt
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.params_dt, n_jobs=self.nb_jobs, cv=self.cv, verbose=1)
        elif algo == 'rf':
            self.learner = RandomForestClassifier(random_state=42, class_weight=balanced_weights)
            self.params_rf = params_rf
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.params_rf, n_jobs=self.nb_jobs, cv=self.cv, verbose=1)
            
    def learn(self, x_train, y_train, x_test, y_test, feature_names, saving_file):
        self.gs_clf.fit(x_train, y_train)
        pred = self.gs_clf.predict(x_test)
        y_train_pred = self.gs_clf.predict(x_train)
        clf_metrics = ClfMetrics()
        train_scores = clf_metrics.score(y_test=y_train, y_pred=y_train_pred)
        train_clf_report = clf_metrics.classif_report(y_test=y_train, y_pred=y_train_pred)
        print(self.learner)
        print('*' * 50)
        print('Train Scores', train_scores)
        test_scores = clf_metrics.score(y_test=y_test, y_pred=pred)
        test_clf_report = clf_metrics.classif_report(y_test=y_test, y_pred=pred)
        print('Test Scores', test_scores)
        print()
        self.saving_dict = defaultdict(dict)
        self.saving_dict['test_scores'] = test_scores
        self.saving_dict['train_metrics'] = train_scores
        self.saving_dict['test_clf_report'] = test_clf_report
        self.saving_dict['train_clf_report'] = train_clf_report
        self.saving_dict['cv_results'] = self.gs_clf.cv_results_
        self.saving_dict['best_params'] = self.gs_clf.best_params_
        self.saving_dict['importances'] = []
        self.saving_dict['rules'] = []
        importances = self.gs_clf.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(100):
            if importances[indices[f]] > 0:
                logger.info("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]],
                                                    feature_names[indices[f]]))
        self.saving_dict['importances'].append(importances)
        self.saving_dict['rules'] = [(f + 1, indices[f], importances[indices[f]], feature_names[indices[f]]) for f in
                            range(100) if importances[indices[f]] > 0]
        
        with open(saving_file, 'w') as fd:
            json.dump(self.saving_dict, fd)
            