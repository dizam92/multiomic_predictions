import numpy as np
import pickle
import logging
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from torch.utils.data import DataLoader
from multiomic_modeling.loss_and_metrics import ClfMetrics
from multiomic_modeling.data.data_loader import MultiomicDataset, SubsetRandomSampler, multiomic_dataset_builder

logging.getLogger('parso.python.diff').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nb_jobs = 16
cv_fold = KFold(n_splits=5)

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

class BaseAlgoTemplate():
    def __init__(self, 
                 algo: str = 'tree', 
                 params_dt: dict = parameters_dt, 
                 params_rf: dict = parameters_rf,
                 nb_jobs: int = nb_jobs, 
                 cv=cv_fold):
        super(BaseAlgoTemplate, self).__init__()
        assert type(algo) == str, 'algo must be either tree, rf or scm'
        assert type(params_dt) == dict, 'params_dt must be a dictionary'
        assert type(parameters_rf) == dict, 'parameters_rf must be a dictionary'
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
        self.saving_dict['importances'] = []
        self.saving_dict['rules'] = []
        self.saving_dict['rules_str'] = []
        importances = self.gs_clf.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(100):
            if importances[indices[f]] > 0:
                logger.info("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]],
                                                    feature_names[indices[f]]))
        self.saving_dict['importances'].append(importances)
        self.saving_dict['rules'] = [(f + 1, indices[f], importances[indices[f]], feature_names[indices[f]]) for f in
                            range(100) if importances[indices[f]] > 0]
        
        with open(saving_file, 'wb') as fd:
            pickle.dump(self.saving_dict, fd)

    @staticmethod
    def reload_dataset(data_size=2000, dataset_views_to_consider='all'): 
        dataset = MultiomicDataset(data_size=data_size, views_to_consider=dataset_views_to_consider)
        train_dataset, test_dataset, valid_dataset = multiomic_dataset_builder(dataset, test_size=0.2, valid_size=0.1)
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
        
        feature_names = train_dataset.dataset.feature_names
        # they are 3D models so i have to reshape them!
        return x_train[:,0,:], y_train, x_test[:,0,:], y_test, feature_names

def run_experiments(data_size: int = 2000, dataset_views_to_consider: str = 'all'):
    dt_base_model = BaseAlgoTemplate(algo='tree')
    rf_base_model = BaseAlgoTemplate(algo='rf')
    x_train, y_train, x_test, y_test, feature_names = dt_base_model.reload_dataset(data_size=data_size, dataset_views_to_consider=dataset_views_to_consider) 
    dt_base_model.learn(x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test, 
                        feature_names=feature_names, 
                        saving_file=f'/home/maoss2/scratch/dt_{dataset_views_to_consider}_data_{data_size}_scores.pck')
    rf_base_model.learn(x_train=x_train, y_train=y_train, 
                        x_test=x_test, y_test=y_test, 
                        feature_names=feature_names, 
                        saving_file=f'/home/maoss2/scratch/rf_{dataset_views_to_consider}_data_{data_size}_scores.pck')


if __name__ == "__main__":
    for view in ['all', 'cnv', 'methyl', 'rna', 'protein', 'mirna']:
        for size in [2000, 5000]:
            run_experiments(data_size=size, dataset_views_to_consider=view)
    # bizare mais mirna n'est pas disponible pour le dernier type de cancer du coup en vaut-t-il la peine de faire cette expérimentation? 
    # Mais c'est du gros n'importe quoi. J'ai pas eu ce pb quand je faisais l'expérience avec optuna! Pk ici ca fait ca? c'est pas possible....
    # run_algo(data_size=743, dataset_views_to_consider='mirna')