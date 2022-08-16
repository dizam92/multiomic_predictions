import numpy as np
import pickle
import logging
import os
import random
from collections import defaultdict
from copy import deepcopy

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from torch.utils.data import DataLoader
from multiomic_modeling.loss_and_metrics import ClfMetrics
from multiomic_modeling.data.data_loader import MultiomicDatasetDataAug, MultiomicDatasetNormal, MultiomicDatasetBuilder, SubsetRandomSampler
from multiomic_modeling.models.base import Model, CustomModelCheckpoint
from multiomic_modeling.torch_utils import get_activation

import torch
import torch.nn as nn

logging.getLogger('parso.python.diff').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nb_jobs = 16
cv_fold = KFold(n_splits=5)

parameters_dt = {'max_depth': np.arange(1, 5),
                 'min_samples_split': np.arange(2, 15),
                 'criterion': ['gini', 'entropy']
                 }
parameters_rf = {'max_depth': np.arange(1, 5),
                 'min_samples_split': np.arange(2, 15),
                 'criterion': ['gini', 'entropy'],
                 'n_estimators': [25, 50, 75, 100]
                 }
parameters_svm = {
    'kernel': ['poly', 'rbf'],
    'C': np.logspace(0.001, 1.0, 5),
    'degree': [2, 3]
}

balanced_weights = {0: 4.03557312, 1: 0.85154295, 2: 0.30184775, 3: 1.18997669, 
                    4: 8.25050505, 5: 0.72372851, 6: 7.73484848, 7: 1.81996435, 
                    8: 0.62294082, 9: 0.61468995,10: 4.07992008, 11: 0.49969411, 
                    12: 1.07615283, 13: 1.85636364, 14: 0.7018388 ,15: 0.84765463, 
                    16: 0.60271547, 17: 0.62398778, 18: 4.26750261, 19: 0.61878788,
                    20: 1.89424861, 21: 1.98541565, 22: 0.65595888, 23: 2.05123054, 
                    24: 1.37001006, 25: 0.77509964, 26: 0.76393565, 27: 2.67102681, 
                    28: 0.64012539, 29: 2.94660895, 30: 0.64012539, 31: 6.51355662, 
                    32: 4.64090909
            }

class BaseAlgoTemplate():
    def __init__(self, 
                 algo: str = 'tree', 
                 params_dt: dict = parameters_dt, 
                 params_rf: dict = parameters_rf,
                 params_svm: dict = parameters_svm, 
                 nb_jobs: int = nb_jobs, 
                 cv=cv_fold):
        super(BaseAlgoTemplate, self).__init__()
        assert type(algo) == str, 'algo must be either tree, rf or scm'
        assert type(params_dt) == dict, 'params_dt must be a dictionary'
        assert type(params_rf) == dict, 'params_rf must be a dictionary'
        assert type(params_svm) == dict, 'params_svm must be a dictionary'
        self.nb_jobs = nb_jobs
        self.cv = cv
        if algo == 'tree':
            self.learner = DecisionTreeClassifier(random_state=42, class_weight=balanced_weights)
            self.params = params_dt
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.params, n_jobs=self.nb_jobs, cv=self.cv, verbose=1)
        elif algo == 'rf':
            self.learner = RandomForestClassifier(random_state=42, class_weight=balanced_weights)
            self.params = params_rf
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.params, n_jobs=self.nb_jobs, cv=self.cv, verbose=1)
        elif algo == 'svm':
            self.learner = SVC(kernel='rbf', gamma='auto', class_weight=balanced_weights, decision_function_shape='ovo', random_state=42)
            self.params = params_svm
            self.gs_clf = GridSearchCV(self.learner, param_grid=self.params, n_jobs=self.nb_jobs, cv=self.cv, verbose=1)
        else: raise ValueError(f'The algoritm {algo} is not supported')
            
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
        self.saving_dict['cv_results'] = self.gs_clf.cv_results_
        self.saving_dict['best_params'] = self.gs_clf.best_params_
        with open(saving_file, 'wb') as fd:
            pickle.dump(self.saving_dict, fd)

    @staticmethod
    def reload_dataset(data_size=2000, dataset_views_to_consider='all', random_state=42): 
        dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider='all')
        train_dataset, test_dataset, valid_dataset = MultiomicDatasetBuilder.multiomic_data_normal_builder(dataset, 
                                                                                                           test_size=0.2,
                                                                                                           valid_size=0.1, 
                                                                                                           random_state=random_state)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        train_dataset_array = next(iter(train_loader))[0][0].numpy()
        train_dataset_array_labels = next(iter(train_loader))[1].numpy()
        
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        x_test = next(iter(test_loader))[0][0].numpy()
        y_test = next(iter(test_loader))[1].numpy()
        
        valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
        valid_dataset_array = next(iter(valid_loader))[0][0].numpy()
        valid_dataset_array_labels = next(iter(valid_loader))[1].numpy()
        
        x_train = np.vstack((train_dataset_array, valid_dataset_array))
        y_train = np.hstack((train_dataset_array_labels, valid_dataset_array_labels))
        # ['all', 'cnv', 'methyl_450', 'mirna', 'rna', 'protein']
        originel_feature_names = train_dataset.dataset.feature_names
        fake_feature_names_mirna = [f'mirna_{idx}' for idx in range(743, 2000)]
        fake_feature_names_protein = [f'protein_{idx}' for idx in range(210, 2000)]
        feature_names = originel_feature_names[:4743] + fake_feature_names_mirna + originel_feature_names[4743:] + fake_feature_names_protein
        
        if dataset_views_to_consider == 'all':
            return x_train.reshape(len(x_train), -1), y_train, x_test.reshape(len(x_test), -1), y_test, feature_names
        if dataset_views_to_consider == 'cnv':
            return x_train.reshape(len(x_train), -1)[:,:2000], y_train, x_test.reshape(len(x_test), -1)[:,:2000], y_test, feature_names[:2000]
        if dataset_views_to_consider == 'methyl':
            return x_train.reshape(len(x_train), -1)[:,2000:4000], y_train, x_test.reshape(len(x_test), -1)[:,2000:4000], y_test, feature_names[2000:4000]
        if dataset_views_to_consider == 'mirna':
            return x_train.reshape(len(x_train), -1)[:,4000:6000], y_train, x_test.reshape(len(x_test), -1)[:,4000:6000], y_test, feature_names[4000:6000]
        if dataset_views_to_consider == 'rna':
            return x_train.reshape(len(x_train), -1)[:,6000:8000], y_train, x_test.reshape(len(x_test), -1)[:,6000:8000], y_test, feature_names[6000:8000]
        if dataset_views_to_consider == 'protein':
            return x_train.reshape(len(x_train), -1)[:,8000:], y_train, x_test.reshape(len(x_test), -1)[:,8000:], y_test, feature_names[8000:]
        

class RunExperiments:
    def __init__(self, 
                 data_size: int = 2000, 
                 dataset_views_to_consider: str = 'all', 
                 random_state: int = 42):
        super().__init__()
        self.dataset_views_to_consider = dataset_views_to_consider
        self.data_size = data_size
        self.random_state = random_state
        
    def run_dt(self):
        dt_base_model = BaseAlgoTemplate(algo='tree')
        x_train, y_train, x_test, y_test, feature_names = dt_base_model.reload_dataset(data_size=self.data_size, 
                                                                                       dataset_views_to_consider=self.dataset_views_to_consider,
                                                                                       random_state=self.random_state) 
        dt_base_model.learn(x_train=x_train, 
                            y_train=y_train,
                            x_test=x_test, 
                            y_test=y_test, 
                            feature_names=feature_names, 
                            saving_file=f'/home/maoss2/scratch/dt_{self.dataset_views_to_consider}_data_{self.data_size}_randomState_{self.random_state}_scores.pck')

    def run_rf(self):
        rf_base_model = BaseAlgoTemplate(algo='rf')
        x_train, y_train, x_test, y_test, feature_names = rf_base_model.reload_dataset(data_size=self.data_size, 
                                                                                       dataset_views_to_consider=self.dataset_views_to_consider,
                                                                                       random_state=self.random_state) 
        rf_base_model.learn(x_train=x_train, 
                            y_train=y_train, 
                            x_test=x_test, 
                            y_test=y_test, 
                            feature_names=feature_names, 
                            saving_file=f'/home/maoss2/scratch/rf_{self.dataset_views_to_consider}_data_{self.data_size}_randomState_{self.random_state}_scores.pck')
    
    def run_svm(self):
        svm_base_model = BaseAlgoTemplate(algo='svm')
        x_train, y_train, x_test, y_test, feature_names = svm_base_model.reload_dataset(data_size=self.data_size, 
                                                                                        dataset_views_to_consider=self.dataset_views_to_consider,
                                                                                       random_state=self.random_state) 
        svm_base_model.learn(x_train=x_train, 
                             y_train=y_train, 
                             x_test=x_test, 
                             y_test=y_test, 
                             feature_names=feature_names, 
                             saving_file=f'/home/maoss2/scratch/svm_{self.dataset_views_to_consider}_data_{self.data_size}_randomState_{self.random_state}_scores.pck')

if __name__ == "__main__":
    data_size = 2000
    view = 'all'
    # seed = 42
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.sample(range(0, 1000), 4)
    # random_seed_list = [42, 78, 433, 966, 699] # that was randomly generate from the precedent code and we add the magic number 42
    random_seed_list = [699]
    for r_s in random_seed_list:
        run_expe_obj = RunExperiments(data_size=data_size, dataset_views_to_consider=view, random_state=r_s)
        if not os.path.exists(f'/home/maoss2/scratch/rf_{view}_data_{data_size}_randomState_{r_s}_scores.pck'): run_expe_obj.run_rf()
        # if not os.path.exists(f'/home/maoss2/scratch/dt_{view}_data_{data_size}_randomState_{r_s}_scores.pck'): run_expe_obj.run_dt()
        if not os.path.exists(f'/home/maoss2/scratch/svm_{view}_data_{data_size}_randomState_{r_s}_scores.pck'): run_expe_obj.run_svm()