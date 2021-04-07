import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from scipy.stats import median_absolute_deviation
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

files_path_on_graham = '/home/maoss2/project/maoss2/tcga_pan_cancer_dataset/data_hdf5'
# files_path_on_graham = '/Users/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/data/tcga_pan_cancer_dataset'
class FichierPath:
    exon_file = f'{files_path_on_graham}/exon_pancan_tcga_reduced.h5'
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga_reduced.h5'
    methyl27_file = f'{files_path_on_graham}/methyl_27_pancan_tcga_reduced.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga_reduced.h5'
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga_reduced.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga_reduced.h5'
    rna_file = f'{files_path_on_graham}/rna_pancan_tcga_reduced.h5'
    rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga_reduced.h5'
    survival_file = f'{files_path_on_graham[:-10]}/Survival_SupplementalTable_S1_20171025_xena_sp'
    # survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    
def read_h5py(fichier):
    d = h5py.File(fichier, 'r')
    data = d['dataset'][()]
    feature_names = np.asarray([el.decode("utf-8") for el in d['features_names'][()]])
    patient_names = np.asarray([el.decode("utf-8") for el in d['patients_names'][()]])
    patient_names = dict(zip(patient_names, np.arange(len(patient_names))))
    return {'data': data, 
            'feature_names': feature_names, 
            'patient_names': patient_names}

def read_pandas_csv(fichier):
    return pd.read_csv(fichier, sep='\t')
    
class MultiomicDataset(Dataset):
    def __init__(self):
        super(MultiomicDataset, self).__init__()
        self.views = [
            read_h5py(fichier=FichierPath.cnv_file), 
            read_h5py(fichier=FichierPath.methyl27_file),
            read_h5py(fichier=FichierPath.methyl450_file),
            read_h5py(fichier=FichierPath.exon_file),
            read_h5py(fichier=FichierPath.mirna_file),
            read_h5py(fichier=FichierPath.rna_file),
            # read_h5py(fichier=FichierPath.rna_iso_file),
            read_h5py(fichier=FichierPath.protein_file)]
        self.nb_features = np.max([view['data'].shape[1] for view in self.views])
        self.survival_data = read_pandas_csv(fichier=FichierPath.survival_file)
        self.sample_to_labels = {self.survival_data['sample'].values[idx]: self.survival_data['cancer type abbreviation'].values[idx] for idx, _ in enumerate(self.survival_data['sample'].values)}
        self.all_patient_names = np.asarray(list(self.sample_to_labels.keys()))
        self.all_patient_labels = np.asarray(list(self.sample_to_labels.values()))
        self.all_patient_labels = LabelEncoder().fit_transform(self.all_patient_labels)
        
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
        return (data.astype(float), mask), patient_label
        
    def __len__(self):
        return len(self.all_patient_names)
        
        
def multiomic_dataset_builder(dataset, test_size=0.2, valid_size=0.1):
    n = len(dataset)
    idxs = np.arange(n)
    labels = dataset.all_patient_labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42) 
    for train_index, test_index in sss.split(idxs, labels):
        train_idx = train_index
        test_idx = test_index
        idxs = idxs[train_index]
        labels = labels[train_index]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=42) 
    for train_index, valid_index in sss.split(idxs, labels):
        train_idx = train_index
        valid_idx = valid_index
        idxs = idxs[train_index]
    train_dataset = Subset(dataset, indices=train_idx)
    test_dataset = Subset(dataset, indices=test_idx)
    valid_dataset =  Subset(dataset, indices=valid_idx)
    return train_dataset, test_dataset, valid_dataset

def multiomic_dataset_loader(dataset, batch_size=32, nb_cpus=2):
    n = len(dataset)
    idx = np.arange(n)
    data_sampler = SubsetRandomSampler(idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, num_workers=nb_cpus)
    return data_loader


