import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import class_weight
from scipy.stats import median_absolute_deviation
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from itertools import combinations

# files_path_on_graham = '/home/maoss2/project/maoss2/tcga_pan_cancer_dataset/data_hdf5'
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
    
class FichierPathCompleteDataset: #matrice complete des 4 vues a 493411 features
    exon_file = f'{files_path_on_graham}/exon_pancan_tcga.h5' 
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga.h5' # (10845, 24776)
    methyl27_file = f'{files_path_on_graham}/methyl_27_pancan_tcga.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga.h5' # (9664, 100000)(-31278) * (9664, 100000)(-31711) * (9664, 100000)(-33085) * (9664, 96065) (-30718 qui sont nan) == 
                                                                        # (9664, 269273) features restant
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga.h5' # (10824, 743)
    rna_file = f'{files_path_on_graham}/rna_pancan_tcga.h5'
    rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga.h5' # (10534, 100000) * (10534, 98619)
    survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    patients_without_view_file = f'{files_path_on_graham}/patients_without_view.txt'
    patients_with_one_view_file = f'{files_path_on_graham}/patients_with_one_view.txt'
    patients_with_two_or_more_views_file = f'{files_path_on_graham}/patients_with_two_or_more_views.txt'
    patients_with_all_4_views_available_file = f'{files_path_on_graham}/patients_with_all_4_views_available.txt'

def read_h5py_all_data(fichier, normalization=False) -> dict:
    d = h5py.File(fichier, 'r')
    if fichier[62:].startswith('cnv') or fichier[62:].startswith('mirna'):
        data = d['dataset_0'][()].T
        feature_names = np.asarray([el.decode("utf-8") for el in d['features_names'][()]])
        patient_names = np.asarray([el.decode("utf-8") for el in d['patients_names'][()]])
        patient_names = patient_names[1:]
        patient_names = dict(zip(patient_names, np.arange(len(patient_names))))
        return {'data': data, 
                'feature_names': feature_names, 
                'patient_names': patient_names}
        
    if fichier[62:].startswith('methyl_450'):
        feature_names = np.asarray([el.decode("utf-8") for el in d['features_names'][()]])
        patient_names = np.asarray([el.decode("utf-8") for el in d['patients_names'][()]])
        patient_names = patient_names[1:]
        patient_names = dict(zip(patient_names, np.arange(len(patient_names))))
        
        data_0 = d['dataset_0'][()].T
        idx_0 = np.unique(np.where(np.isnan(data_0))[1])
        feature_names_to_be_deleted_data = feature_names[idx_0]
        feature_names = np.asarray(list(set(feature_names) - set(feature_names_to_be_deleted_data)))
        data_0 = data_0[:, ~np.isnan(data_0).any(axis=0)]

        data_1 = d['dataset_1'][()].T
        idx_1 = np.unique(np.where(np.isnan(data_1))[1])
        feature_names_to_be_deleted_data = feature_names[idx_1]
        feature_names = np.asarray(list(set(feature_names) - set(feature_names_to_be_deleted_data)))
        data_1 = data_1[:, ~np.isnan(data_1).any(axis=0)]

        data_2 = d['dataset_2'][()].T
        idx_2 = np.unique(np.where(np.isnan(data_2))[1])
        feature_names_to_be_deleted_data = feature_names[idx_2]
        feature_names = np.asarray(list(set(feature_names) - set(feature_names_to_be_deleted_data)))
        data_2 = data_2[:, ~np.isnan(data_2).any(axis=0)]
        
        data_3 = d['dataset_3'][()].T
        idx_3 = np.unique(np.where(np.isnan(data_3))[1])
        feature_names_to_be_deleted_data = feature_names[idx_3]
        feature_names = np.asarray(list(set(feature_names) - set(feature_names_to_be_deleted_data)))
        data_3 = data_3[:, ~np.isnan(data_3).any(axis=0)]
        
        data = np.hstack((data_0, data_1, data_2, data_3))
        return {'data': data,
                'feature_names': feature_names, 
                'patient_names': patient_names}
        
    if fichier[62:].startswith('rna_isoforms'):
        feature_names = np.asarray([el.decode("utf-8") for el in d['features_names'][()]])
        patient_names = np.asarray([el.decode("utf-8") for el in d['patients_names'][()]])
        patient_names = patient_names[1:]
        patient_names = dict(zip(patient_names, np.arange(len(patient_names))))
        
        data_0 = d['dataset_0'][()].T
        idx_0 = np.unique(np.where(np.isnan(data_0))[1])
        feature_names_to_be_deleted_data = feature_names[idx_0]
        feature_names = np.asarray(list(set(feature_names) - set(feature_names_to_be_deleted_data)))
        data_0 = data_0[:, ~np.isnan(data_0).any(axis=0)]

        data_1 = d['dataset_1'][()].T
        idx_1 = np.unique(np.where(np.isnan(data_1))[1])
        feature_names_to_be_deleted_data = feature_names[idx_1]
        feature_names = np.asarray(list(set(feature_names) - set(feature_names_to_be_deleted_data)))
        data_1 = data_1[:, ~np.isnan(data_1).any(axis=0)]
        
        data = np.hstack((data_0, data_1))
        return {'data': data, 
                'feature_names': feature_names, 
                'patient_names': patient_names}
            
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

class MultiomicDataset(Dataset):
    def __init__(self, data_size=2000, views_to_consider='all', type_of_model='transformer', complete_dataset=False):
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
        if views_to_consider == 'all':
            if complete_dataset:
                self.views = [
                read_h5py_all_data(fichier=FichierPathCompleteDataset.cnv_file, normalization=False), 
                read_h5py_all_data(fichier=FichierPathCompleteDataset.methyl450_file, normalization=False),
                read_h5py_all_data(fichier=FichierPathCompleteDataset.mirna_file, normalization=True),
                read_h5py_all_data(fichier=FichierPathCompleteDataset.rna_iso_file, normalization=True)
            ]
            else:
                self.views = [
                    read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                    read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                    read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                    read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
                ]
        elif views_to_consider == 'cnv':
            if complete_dataset:
                self.views = [
                read_h5py_all_data(fichier=FichierPathCompleteDataset.cnv_file, normalization=False)
            ]
            else:
                self.views = [
                    read_h5py(fichier=FichierPath.cnv_file, normalization=False)
                ]
        elif views_to_consider == 'methyl':
            if complete_dataset:
                self.views = [
                read_h5py_all_data(fichier=FichierPathCompleteDataset.methyl450_file, normalization=False)
            ]
            else:
                self.views = [
                    read_h5py(fichier=FichierPath.methyl450_file, normalization=False)
                ]
        elif views_to_consider == 'exon':
            if complete_dataset:
                self.views = [
                read_h5py_all_data(fichier=FichierPathCompleteDataset.exon_file, normalization=True)
            ]
            else:
                self.views = [
                    read_h5py(fichier=FichierPath.exon_file, normalization=True)
                ]
        elif views_to_consider == 'mirna':
            if complete_dataset:
                    self.views = [
                    read_h5py_all_data(fichier=FichierPathCompleteDataset.mirna_file, normalization=True)
                ]
            else:
                self.views = [
                    read_h5py(fichier=FichierPath.mirna_file, normalization=True)
                ]
        elif views_to_consider == 'rna':
            if complete_dataset:
                self.views = [
                read_h5py_all_data(fichier=FichierPathCompleteDataset.rna_file, normalization=True)
            ]
            else:
                self.views = [
                    read_h5py(fichier=FichierPath.rna_file, normalization=True)
                ]
        elif views_to_consider == 'rna_iso':
            if complete_dataset:
                self.views = [
                read_h5py_all_data(fichier=FichierPathCompleteDataset.rna_iso_file, normalization=True)
            ]
            else:
                self.views = [
                    read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
                ]
        elif views_to_consider == 'protein':
            if complete_dataset:
                self.views = [
                read_h5py_all_data(fichier=FichierPathCompleteDataset.protein_file, normalization=True)
            ]
            else:
                self.views = [
                    read_h5py(fichier=FichierPath.protein_file, normalization=True)
                ]
        elif views_to_consider == 'cnv_methyl_rna':
            self.views = [
                    read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                    read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                    read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
                ]
        elif views_to_consider == 'cnv_methyl_mirna':
             self.views = [
                    read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                    read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                    read_h5py(fichier=FichierPath.mirna_file, normalization=True)
                ]
        elif views_to_consider == 'methyl_mirna_rna':
            self.views = [
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif views_to_consider == 'cnv_mirna_rna':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif views_to_consider == 'cnv_mirna':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False),
                read_h5py(fichier=FichierPath.mirna_file, normalization=True)
            ]
        elif views_to_consider == 'cnv_rna':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif views_to_consider == 'cnv_methyl':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False)
            ]
        elif views_to_consider == 'mirna_rna':
            self.views = [
                read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif views_to_consider == 'methyl_mirna':
            self.views = [
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                read_h5py(fichier=FichierPath.mirna_file, normalization=True)
            ]
        elif views_to_consider == 'methyl_rna':
            self.views = [
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        else:
            raise ValueError(f'the view {views_to_consider} is not available in the dataset')
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
        elif views_to_consider in ['cnv', 'methyl', 'exon', 'mirna', 'rna', 'rna_iso', 'protein']:
            patients_name_view = list(self.views[0]['patient_names'].keys())
            for patient_name in list(self.sample_to_labels.keys()):
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        elif views_to_consider in ['cnv_methyl_rna', 'cnv_methyl_mirna', 'cnv_mirna_rna', 'methyl_mirna_rna', 'cnv_mirna', 
                                   'cnv_rna', 'cnv_methyl', 'mirna_rna', 'methyl_mirna','methyl_rna']:
            patients_name_views = []
            for idx in range(len(self.views)):
                patients_name_views.extend(list(self.views[idx]['patient_names'].keys()))
            patients_name_views = list(np.unique(patients_name_views))
            for patient_name in list(self.sample_to_labels.keys()):
                if patient_name not in patients_name_views:
                    self.sample_to_labels.pop(patient_name)
        else:
            raise ValueError(f'the view {views_to_consider} is not available in the dataset')
        self.all_patient_names = np.asarray(list(self.sample_to_labels.keys()))
        self.all_patient_labels = np.asarray(list(self.sample_to_labels.values()))
        self.all_patient_labels = LabelEncoder().fit_transform(self.all_patient_labels)
        self.type_of_model = type_of_model
        self.class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(self.all_patient_labels),
                                                 self.all_patient_labels) #pylint deconne sinon pas d'erreurs
        self.number_of_views = len(self.views)
        self.sample_to_labels = {self.all_patient_names[idx]: self.all_patient_labels[idx] 
                                 for idx, _ in enumerate(self.all_patient_names)}
        self.all_data_combination = self.build_combination_per_examples()
               
    def build_combination_per_examples(self):
        data_list = []
        if self.number_of_views == 1:
            for patient_name, patient_label in self.sample_to_labels.items():
                data = np.zeros((self.number_of_views, self.nb_features))
                for i, view in enumerate(self.views):
                    if patient_name in view['patient_names']:
                        try:
                            data[i] = view['data'][view['patient_names'].get(patient_name, 0)]
                        except ValueError:
                            data[i][:view['data'][view['patient_names'].get(patient_name, 0)].shape[0]] = view['data'][view['patient_names'].get(patient_name, 0)]
                mask = np.array([(patient_name in view['patient_names']) for view in self.views])
                data_list.append([patient_name, patient_label, data.astype(float), mask])
        else:
            for patient_name, patient_label in self.sample_to_labels.items():
                idx_views_combinations = []
                for pos in range(2, self.number_of_views + 1): 
                    idx_views_combinations.extend(MultiomicDataset.rSubset(arr=np.arange(self.number_of_views), r=pos))
                for comb in idx_views_combinations:
                    # data = np.zeros((len(comb), self.nb_features)) # nombre_views X nombre_features
                    data = np.zeros((self.number_of_views, self.nb_features))
                    for idx in range(len(comb)): # equivalent à un vecteur allant de [0 à data.shape[0]]
                        try:
                            data[idx] = self.views[comb[idx]]['data'][self.views[comb[idx]]['patient_names'].get(patient_name, 0)]
                        except ValueError:
                            data[idx][:self.views[comb[idx]]['data'][self.views[comb[idx]]['patient_names'].get(patient_name, 0)].shape[0]] = self.views[comb[idx]]['data'][self.views[comb[idx]]['patient_names'].get(patient_name, 0)]
                    mask = np.array([(patient_name in view['patient_names']) for view in self.views])
                    # mask = np.array([(patient_name in self.views[comb[idx]]['patient_names']) for idx in range(len(comb))])
                    # assert len(mask) == data.shape[0], 'Something went wrong with the mask size and the inner data.shape'
                    data_list.append([patient_name, patient_label, data.astype(float), mask])
        return data_list
    
    def __getitem__(self, idx):
        data = np.asarray(self.all_data_combination[idx][2])
        mask = np.asarray(self.all_data_combination[idx][3])
        patient_label = self.all_data_combination[idx][1]
        return (data, mask), patient_label
        
    def __len__(self):
        # return len(self.all_patient_names)
        return len(self.all_data_combination)
        
    @staticmethod
    def rSubset(arr, r):
        """
        return list of all subsets of length r; to deal with duplicate subsets use or just use return list(combinations(arr, r))
        """
        return list(set(list(combinations(arr, r))))
    
def multiomic_dataset_builder(dataset, test_size=0.2, valid_size=0.1):
    patient_names = dataset.all_patient_names
    labels = dataset.all_patient_labels
    X_train, X_test, y_train, y_test = train_test_split(patient_names, labels, test_size=test_size, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=42)
    X_train_indices = []; X_test_indices = []; X_valid_indices = []
    for idx, example in enumerate(dataset.all_data_combination): # Example : (patient_name, patient_label, data.astype(float), np.array(mask))
        if example[0] in X_train: X_train_indices.append(idx)
        if example[0] in X_test: X_test_indices.append(idx)
        if example[0] in X_valid: X_valid_indices.append(idx)
        
    train_dataset = Subset(dataset, indices=X_train_indices)
    test_dataset = Subset(dataset, indices=X_test_indices)
    valid_dataset =  Subset(dataset, indices=X_valid_indices)
    return train_dataset, test_dataset, valid_dataset

def multiomic_dataset_loader(dataset, batch_size=32, nb_cpus=2):
    n = len(dataset)
    idx = np.arange(n)
    data_sampler = SubsetRandomSampler(idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, num_workers=nb_cpus)
    return data_loader


    # TODO: ORIGINAL TO NOT DELETE 
    # def __getitem__(self, idx): 
    #     patient_name = self.all_patient_names[idx]
    #     patient_label = self.all_patient_labels[idx]
    #     data = np.zeros((len(self.views), self.nb_features)) # nombre_views X nombre_features 
    #     for i, view in enumerate(self.views):
    #         if patient_name in view['patient_names']:
    #             try:
    #                 data[i] = view['data'][view['patient_names'].get(patient_name, 0)]
    #             except ValueError:
    #                 data[i][:view['data'][view['patient_names'].get(patient_name, 0)].shape[0]] = view['data'][view['patient_names'].get(patient_name, 0)]
    #     mask = np.array([(patient_name in view['patient_names']) for view in self.views])
    #     if self.type_of_model == 'transformer':
    #         return (data.astype(float), mask), patient_label
    #     if self.type_of_model == 'mlp':
    #         data = data.reshape(-1)
    #         return data.astype(float), patient_label
    
    # TODO: The one that doing the job in the getitem but i cant visualize how to use them downstream. DONT DELETE    
    # def __getitem__(self, idx):
    #     patient_name = self.all_patient_names[idx]
    #     patient_label = self.all_patient_labels[idx]
    #     data = np.zeros((len(self.views), self.nb_features)) # nombre_views X nombre_features 
    #     idx_views_combinations = []
    #     data_list = []
    #     mask = np.array([(patient_name in view['patient_names']) for view in self.views]) # le meme mask va etre réutiliser: par défaut je garde la structure de base (4 views)
    #     for pos in range(2, self.number_of_views + 1): idx_views_combinations.extend(MultiomicDataset.rSubset(arr=np.arange(4), r=pos))
    #     for comb in idx_views_combinations:
    #         data = np.zeros((len(self.views), self.nb_features))
    #         for idx in comb:
    #             try:
    #                 data[idx] = self.views[idx]['data'][self.views[idx]['patient_names'].get(patient_name, 0)]
    #             except ValueError:
    #                 data[idx][:self.views[idx]['data'][self.views[idx]['patient_names'].get(patient_name, 0)].shape[0]] = self.views[idx]['data'][self.views[idx]['patient_names'].get(patient_name, 0)]
    #         data_list.append((data.astype(float), mask))
    #     return data_list, patient_label
        
# TODO: ORIGINAL TO NOT DELETE     
# def multiomic_dataset_builder(dataset, test_size=0.2, valid_size=0.1):
#     n = len(dataset)
#     idxs = np.arange(n)
#     labels = dataset.all_patient_labels
#     X_train, X_test, y_train, y_test = train_test_split(idxs, labels, test_size=test_size, random_state=42)
#     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=42)
#     train_dataset = Subset(dataset, indices=X_train)
#     test_dataset = Subset(dataset, indices=X_test)
#     valid_dataset =  Subset(dataset, indices=X_valid)
#     return train_dataset, test_dataset, valid_dataset
