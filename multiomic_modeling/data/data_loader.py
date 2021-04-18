import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import class_weight
from scipy.stats import median_absolute_deviation
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

# files_path_on_graham = '/home/maoss2/project/maoss2/tcga_pan_cancer_dataset/data_hdf5'
files_path_on_graham = '/Users/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/data/tcga_pan_cancer_dataset'
class FichierPath:
    exon_file = f'{files_path_on_graham}/exon_pancan_tcga_reduced.h5'
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga_reduced.h5'
    methyl27_file = f'{files_path_on_graham}/methyl_27_pancan_tcga_reduced.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga_reduced.h5'
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga_reduced.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga_reduced.h5'
    rna_file = f'{files_path_on_graham}/rna_pancan_tcga_reduced.h5'
    rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga_reduced.h5'
    survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    patients_without_view_file = f'{files_path_on_graham}/patients_without_view.txt'
    patients_with_one_view_file = f'{files_path_on_graham}/patients_with_one_view.txt'
    
def read_h5py(fichier, normalization=False) -> dict:
    d = h5py.File(fichier, 'r')
    data = d['dataset'][()]
    if normalization:
        data = StandardScaler().fit_transform(data)
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

patients_without_view = read_file_txt(FichierPath.patients_without_view_file)
patients_with_one_view_file = read_file_txt(FichierPath.patients_with_one_view_file)

class MultiomicDataset(Dataset):
    def __init__(self, views_to_consider='all'):
        super(MultiomicDataset, self).__init__()
        """
        Arguments:
            views_to_consider, str, 
                all, load all the 8 views (cnv, methyl27, methyl450, exon, mirna, rna, rna_iso, protein)
                cnv, load just cnv views
                methyl, load just methyl27 and methyl450 views
                exon, load just exon views
                mirna, load just mirna views
                rna, load just rna views
                rna_iso, load just rna_iso views
                protein, load just protein views
        """
        if views_to_consider == 'all':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                read_h5py(fichier=FichierPath.methyl27_file, normalization=False),
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                read_h5py(fichier=FichierPath.exon_file, normalization=True),
                read_h5py(fichier=FichierPath.mirna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_file, normalization=True),
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True),
                read_h5py(fichier=FichierPath.protein_file, normalization=True)
            ]
        elif views_to_consider == 'cnv':
            self.views = [
                read_h5py(fichier=FichierPath.cnv_file, normalization=False)
            ]
        elif views_to_consider == 'methyl':
            self.views = [
                read_h5py(fichier=FichierPath.methyl27_file, normalization=False),
                read_h5py(fichier=FichierPath.methyl450_file, normalization=False)
            ]
        elif views_to_consider == 'exon':
            self.views = [
                read_h5py(fichier=FichierPath.exon_file, normalization=True)
            ]
        elif views_to_consider == 'mirna':
            self.views = [
                read_h5py(fichier=FichierPath.mirna_file, normalization=True)
            ]
        elif views_to_consider == 'rna':
            self.views = [
                read_h5py(fichier=FichierPath.rna_file, normalization=True)
            ]
        elif views_to_consider == 'rna_iso':
            self.views = [
                read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif views_to_consider == 'protein':
            self.views = [
                read_h5py(fichier=FichierPath.protein_file, normalization=True)
            ]
        else:
            raise ValueError(f'the view {views_to_consider} is not available in the dataset')
        self.nb_features = np.max([view['data'].shape[1] for view in self.views])
        self.survival_data = read_pandas_csv(fichier=FichierPath.survival_file)
        self.sample_to_labels = {self.survival_data['sample'].values[idx]: self.survival_data['cancer type abbreviation'].values[idx] for idx, _ in enumerate(self.survival_data['sample'].values)}
        for patient_name in patients_without_view:
            self.sample_to_labels.pop(patient_name)
        if views_to_consider == 'all':
            for patient_name in patients_with_one_view_file:
                self.sample_to_labels.pop(patient_name)        
        elif views_to_consider == 'cnv':
            patients_name_view = list(self.views[0]['patient_names'].keys())
            for patient_name in self.sample_to_labels.keys():
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        elif views_to_consider == 'methyl':
            patients_name_methyl_views = []
            patients_name_methyl_views.extend(list(self.views[0]['patient_names'].keys()))
            patients_name_methyl_views.extend(list(self.views[1]['patient_names'].keys()))
            for patient_name in self.sample_to_labels.keys():
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        elif views_to_consider == 'exon':
            patients_name_view = list(self.views[0]['patient_names'].keys())
            for patient_name in self.sample_to_labels.keys():
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        elif views_to_consider == 'mirna':
            patients_name_view = list(self.views[0]['patient_names'].keys())
            for patient_name in self.sample_to_labels.keys():
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        elif views_to_consider == 'rna':
            patients_name_view = list(self.views[0]['patient_names'].keys())
            for patient_name in self.sample_to_labels.keys():
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        elif views_to_consider == 'rna_iso':
            patients_name_view = list(self.views[0]['patient_names'].keys())
            for patient_name in self.sample_to_labels.keys():
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        elif views_to_consider == 'protein':
            patients_name_view = list(self.views[0]['patient_names'].keys())
            for patient_name in self.sample_to_labels.keys():
                if patient_name not in patients_name_view:
                    self.sample_to_labels.pop(patient_name) 
        else:
            raise ValueError(f'the view {views_to_consider} is not available in the dataset')
        self.all_patient_names = np.asarray(list(self.sample_to_labels.keys()))
        self.all_patient_labels = np.asarray(list(self.sample_to_labels.values()))
        self.all_patient_labels = LabelEncoder().fit_transform(self.all_patient_labels)
        # self.class_weights = class_weight.compute_class_weight('balanced',
        #                                          np.unique(self.all_patient_labels),
        #                                          self.all_patient_labels) #pylint deconne sinon pas d'erreurs
        
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
        # data = data.reshape(-1) # 16000
        # return data.astype(float), patient_label
        return (data.astype(float), mask), patient_label
        
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
    return data_loader


