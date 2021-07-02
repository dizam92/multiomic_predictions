import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import class_weight
from copy import deepcopy
from scipy.stats import median_absolute_deviation
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from itertools import combinations

files_path_on_graham = '/project/6000474/maoss2/tcga_pan_cancer_dataset/data_hdf5'
# files_path_on_graham = '/Users/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/data/tcga_pan_cancer_dataset'
class FichierPath:
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga_reduced_2000.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga_reduced_2000.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga_reduced_2000.h5'
    rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga_reduced_2000.h5'
    exon_file = f'{files_path_on_graham}/exon_pancan_tcga_reduced.h5'
    methyl27_file = f'{files_path_on_graham}/methyl_27_pancan_tcga_reduced.h5'
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga_reduced.h5'
    rna_file = f'{files_path_on_graham}/rna_pancan_tcga_reduced.h5'
    survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    patients_without_view_file = f'{files_path_on_graham}/patients_without_view.txt'
    patients_with_one_view_file = f'{files_path_on_graham}/patients_with_one_view.txt'
    patients_with_two_or_more_views_file = f'{files_path_on_graham}/patients_with_two_or_more_views.txt'
    patients_with_all_4_views_available_file = f'{files_path_on_graham}/patients_with_all_4_views_available.txt'

class FichierPath5K:
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga_reduced_5000.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga_reduced_5000.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga_reduced_5000.h5'
    rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga_reduced_5000.h5'
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga_reduced.h5'
    exon_file = f'{files_path_on_graham}/exon_pancan_tcga_reduced.h5'
    survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    patients_without_view_file = f'{files_path_on_graham}/patients_without_view.txt'
    patients_with_one_view_file = f'{files_path_on_graham}/patients_with_one_view.txt'
    patients_with_two_or_more_views_file = f'{files_path_on_graham}/patients_with_two_or_more_views.txt'
    patients_with_all_4_views_available_file = f'{files_path_on_graham}/patients_with_all_4_views_available.txt'

class FichierPath10K:
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga_reduced_10000.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga_reduced_10000.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga_reduced_10000.h5'
    rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga_reduced_10000.h5'
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga_reduced.h5'
    exon_file = f'{files_path_on_graham}/exon_pancan_tcga_reduced.h5'
    survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    patients_without_view_file = f'{files_path_on_graham}/patients_without_view.txt'
    patients_with_one_view_file = f'{files_path_on_graham}/patients_with_one_view.txt'
    patients_with_two_or_more_views_file = f'{files_path_on_graham}/patients_with_two_or_more_views.txt'
    patients_with_all_4_views_available_file = f'{files_path_on_graham}/patients_with_all_4_views_available.txt'
    
def read_h5py(fichier, normalization=False) -> dict:
    d = h5py.File(fichier, 'r')
    data = d['dataset'][()]
    if normalization:
        data = MinMaxScaler().fit_transform(data)
    feature_names = np.asarray([el.decode("utf-8") for el in d['features_names'][()]])
    patient_names = np.asarray([el.decode("utf-8") for el in d['patients_names'][()]])
    patient_names = dict(zip(patient_names, np.arange(len(patient_names))))
    return {'data': data, 
            'feature_names': feature_names, 
            'patient_names': patient_names}

def read_pandas_csv(fichier):
    return pd.read_csv(fichier, sep='\t')

def read_file_txt(fichier) -> list:
    with open(fichier, 'r') as f:
        lines = [l.strip('\n') for l in f.readlines()] 
    return lines

def clean_patients_list_problem():
    """
    Utility function used to build the file where all the aptients have at least 2 views in all the 4 
    [cnv, methyl450, mirna, rna_iso]
    """
    cnv = read_h5py(fichier=FichierPath.cnv_file, normalization=False)
    methyl = read_h5py(fichier=FichierPath.methyl450_file, normalization=False)
    mirna = read_h5py(fichier=FichierPath.mirna_file, normalization=True)
    rna = read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
    survival_data = read_pandas_csv(fichier=FichierPath.survival_file)
    sample_to_labels = {survival_data['sample'].values[idx]: survival_data['cancer type abbreviation'].values[idx] 
                        for idx, _ in enumerate(survival_data['sample'].values)}
    sample_to_labels_keys = list(sample_to_labels.keys())  
    patients_cnv = list(cnv['patient_names'].keys())
    patients_methyl = list(methyl['patient_names'].keys())
    patients_mirna = list(mirna['patient_names'].keys())
    patients_rna = list(rna['patient_names'].keys())
    cnv_inter_methyl = list(set(patients_cnv).intersection(set(patients_methyl)))
    cnv_inter_mirna = list(set(patients_cnv).intersection(set(patients_mirna)))
    cnv_inter_rna = list(set(patients_cnv).intersection(set(patients_rna)))
    methyl_inter_mirna = list(set(patients_methyl).intersection(set(patients_mirna)))
    methyl_inter_rna = list(set(patients_methyl).intersection(set(patients_rna)))
    mirna_inter_rna = list(set(patients_mirna).intersection(set(patients_rna)))
    patients_with_two_or_more_views = []
    for patient in sample_to_labels_keys:
        cpt = 0
        if patient in cnv_inter_methyl: cpt += 1
        if patient in cnv_inter_mirna: cpt += 1
        if patient in cnv_inter_rna: cpt += 1
        if patient in methyl_inter_mirna: cpt += 1
        if patient in methyl_inter_rna: cpt += 1
        if patient in mirna_inter_rna: cpt += 1
        if cpt >= 2 :
            patients_with_two_or_more_views.append(patient)
    with open('patients_with_two_or_more_views.txt', 'w') as f:
        for patient in patients_with_two_or_more_views:
            f.write(f'{patient}\n')
    patients_with_all_4_views_available = list(set(sample_to_labels_keys).intersection(set(patients_cnv)).intersection(set(patients_methyl)).intersection(set(patients_mirna)).intersection(set(patients_rna)))
    with open('patients_with_all_4_views_available.txt', 'w') as f:
        for patient in patients_with_all_4_views_available:
            f.write(f'{patient}\n')
   
patients_without_view = read_file_txt(FichierPath.patients_without_view_file)
patients_with_one_view_file = read_file_txt(FichierPath.patients_with_one_view_file)
patients_with_two_or_more_views_file = read_file_txt(FichierPath.patients_with_two_or_more_views_file)
patients_with_all_4_views_available_file = read_file_txt(FichierPath.patients_with_all_4_views_available_file)

class BuildViews(object):
    def __init__(self, data_size, view_name):
        super(BuildViews, self).__init__()
        if data_size == 2000: pass
        if data_size == 5000:
            FichierPath.cnv_file = FichierPath5K.cnv_file
            FichierPath.methyl450_file = FichierPath5K.methyl450_file
            FichierPath.mirna_file = FichierPath5K.mirna_file
            FichierPath.rna_iso_file = FichierPath5K.rna_iso_file
        if data_size == 10000:
            FichierPath.cnv_file = FichierPath10K.cnv_file
            FichierPath.methyl450_file = FichierPath10K.methyl450_file
            FichierPath.mirna_file = FichierPath10K.mirna_file
            FichierPath.rna_iso_file = FichierPath10K.rna_iso_file
        if data_size not in [743, 2000, 5000, 10000]: raise ValueError(f'the data size {data_size} is not available in the dataset')
        if view_name == 'all':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif view_name == 'cnv':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False)
            ]
        elif view_name == 'methyl':
            self.views = [
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False)
            ]
        elif view_name == 'mirna':
            self.views = [
                read_h5py(fichier=FichierPath.mirna_file, normalization=True)
            ]
        elif view_name == 'rna_iso':
            self.views = [
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif view_name == 'cnv_methyl_rna':
            self.views = [
                    read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                    read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                    read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
                ]
        elif view_name == 'cnv_methyl_mirna':
             self.views = [
                    read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                    read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                    read_h5py(fichier=FichierPath.mirna_file, normalization=True)
                ]
        elif view_name == 'methyl_mirna_rna':
            self.views = [
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif view_name == 'cnv_mirna_rna':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif view_name == 'cnv_mirna':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False),
                read_h5py(fichier=FichierPath.mirna_file, normalization=True)
            ]
        elif view_name == 'cnv_rna':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif view_name == 'cnv_methyl':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False)
            ]
        elif view_name == 'mirna_rna':
            self.views = [
                read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif view_name == 'methyl_mirna':
            self.views = [
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                read_h5py(fichier=FichierPath.mirna_file, normalization=True)
            ]
        elif view_name == 'methyl_rna':
            self.views = [
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        else:
            raise ValueError(f'The view {view_name} is not available in the dataset')
        
class MultiomicDataset(Dataset):
    def __init__(self, data_size=2000, views_to_consider='all'):
        super(MultiomicDataset, self).__init__()
        """
        Arguments:
            data_size: int, 2k; 5k or 10k for the specific patch file to load
            views_to_consider, str, 
                all, load all the 4 views (cnv, methyl450, mirna, rna_iso )
                cnv, load just cnv views
                methyl, load just methyl27 and methyl450 views
                exon, load just exon views
                mirna, load just mirna views
                rna, load just rna views
                rna_iso, load just rna_iso views
                protein, load just protein views
            type_of_model: mlp or transformer (impact the get function to have it concatenated or not)
            complete_dataset: to load the original view (complete version) or the selected features version
        """
        self.views = BuildViews(data_size=data_size, view_name=views_to_consider).views
        if views_to_consider == 'mirna': self.nb_features = data_size
        else: self.nb_features = np.max([view['data'].shape[1] for view in self.views])
        self.feature_names  = []
        for view in self.views:
            self.feature_names.extend(list(view['feature_names']))        
        self.survival_data = read_pandas_csv(fichier=FichierPath.survival_file)
        self.sample_to_labels = {self.survival_data['sample'].values[idx]: self.survival_data['cancer type abbreviation'].values[idx] 
                                 for idx, _ in enumerate(self.survival_data['sample'].values)}
        for patient_name in patients_without_view:
            self.sample_to_labels.pop(patient_name)
        if views_to_consider == 'all':
            for patient_name in list(self.sample_to_labels.keys()):
                if patient_name not in patients_with_two_or_more_views_file:
                    self.sample_to_labels.pop(patient_name)    
        elif views_to_consider in ['cnv', 'methyl', 'mirna', 'rna_iso']:
            patients_name_view = list(self.views[0]['patient_names'].keys())
            for patient_name in list(self.sample_to_labels.keys()):
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        elif views_to_consider in ['cnv_methyl_rna', 'cnv_methyl_mirna', 'cnv_mirna_rna', 'methyl_mirna_rna', 
                                   'cnv_mirna', 'cnv_rna', 'cnv_methyl', 'mirna_rna', 'methyl_mirna','methyl_rna']:
            patients_name_views = []
            for idx in range(len(self.views)):
                patients_name_views.extend(list(self.views[idx]['patient_names'].keys()))
            patients_name_views = list(np.unique(patients_name_views))
            for patient_name in list(self.sample_to_labels.keys()):
                if patient_name not in patients_name_views:
                    self.sample_to_labels.pop(patient_name)
        else:
            raise ValueError(f'The view {views_to_consider} is not available in the dataset')
        self.all_patient_names = np.asarray(list(self.sample_to_labels.keys()))
        self.all_patient_labels = np.asarray(list(self.sample_to_labels.values()))
        self.label_encoder = LabelEncoder() # i will need this to inverse_tranform afterward i think for the analysis downstream
        self.all_patient_labels = self.label_encoder.fit_transform(self.all_patient_labels)
        self.class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(self.all_patient_labels),
                                                 self.all_patient_labels) #pylint deconne sinon pas d'erreurs

    def __getitem__(self, idx):
        patient_name = self.all_patient_names[idx]
        patient_label = self.all_patient_labels[idx]
        data = np.zeros((len(self.views), self.nb_features)) # nombre_views X nombre_features
        for i, view in enumerate(self.views):
            if patient_name in view['patient_names']:
                try:
                    data[i] = view['data'][view['patient_names'].get(patient_name, 0)]
                except ValueError:
                    data[i][:view['data'][view['patient_names'].get(patient_name, 0)].shape[0]] = view['data'][view['patient_names'].get(patient_name, 0)]
        mask = np.array([(patient_name in view['patient_names']) for view in self.views])
        original_mask = deepcopy(mask)
        nb_views = np.sum(mask)
        if nb_views > 1:
            n_views_to_drop = np.random.choice(nb_views-1)
            if n_views_to_drop >= 1:
                mask[np.random.choice(np.flatnonzero(mask), size=n_views_to_drop)] = 0
        return (data.astype(float), mask, original_mask), patient_label
    
    def __len__(self):
        return len(self.all_patient_names)
               
def multiomic_dataset_builder(dataset, test_size=0.2, valid_size=0.1):
    n = len(dataset)
    idxs = np.arange(n)
    labels = dataset.all_patient_labels
    X_train, X_test, y_train, y_test = train_test_split(idxs, labels, test_size=test_size, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=42)
    train_dataset = Subset(dataset, indices=X_train)
    test_dataset = Subset(dataset, indices=X_test)
    valid_dataset =  Subset(dataset, indices=X_valid)
    return train_dataset, test_dataset, valid_dataset

def multiomic_dataset_loader(dataset, batch_size=32, nb_cpus=2):
    n = len(dataset)
    idx = np.arange(n)
    data_sampler = SubsetRandomSampler(idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, num_workers=nb_cpus)
    return data_loader #next(iter(test_data))[0]