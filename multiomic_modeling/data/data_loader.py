import pandas as pd
import numpy as np
import h5py
import math
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import class_weight, compute_class_weight
from copy import deepcopy
from scipy.stats import median_absolute_deviation
import torch
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from itertools import combinations

files_path_on_graham = '/project/6000474/maoss2/tcga_pan_cancer_dataset/data_hdf5'
class FichierPath:
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga_reduced_2000.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga_reduced_2000.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga_reduced_2000.h5'
    rna_file = f'{files_path_on_graham}/rna_pancan_tcga_reduced_2000.h5'
    rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga_reduced_2000.h5'
    # exon_file = f'{files_path_on_graham}/exon_pancan_tcga_reduced.h5'
    # methyl27_file = f'{files_path_on_graham}/methyl_27_pancan_tcga_reduced.h5'
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga_reduced_2000.h5'
    survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    patients_without_view_file = f'{files_path_on_graham}/patients_a_exclure_car_sans_vues.txt'
    # patients_with_one_view_file = f'{files_path_on_graham}/patients_with_one_view.txt'
    # patients_with_two_or_more_views_file = f'{files_path_on_graham}/patients_with_two_or_more_views.txt'
    # patients_with_all_4_views_available_file = f'{files_path_on_graham}/patients_with_all_4_views_available.txt'

class FichierPath5K:
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga_reduced_5000.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga_reduced_5000.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga_reduced_5000.h5'
    rna_file = f'{files_path_on_graham}/rna_pancan_tcga_reduced_5000.h5'
    # rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga_reduced_5000.h5'
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga_reduced_5000.h5'
    exon_file = f'{files_path_on_graham}/exon_pancan_tcga_reduced.h5'
    survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    patients_without_view_file = f'{files_path_on_graham}/patients_a_exclure_car_sans_vues.txt'
    # patients_with_one_view_file = f'{files_path_on_graham}/patients_with_one_view.txt'
    # patients_with_two_or_more_views_file = f'{files_path_on_graham}/patients_with_two_or_more_views.txt'
    # patients_with_all_4_views_available_file = f'{files_path_on_graham}/patients_with_all_4_views_available.txt'

class FichierPath10K:
    cnv_file = f'{files_path_on_graham}/cnv_pancan_tcga_reduced_10000.h5'
    methyl450_file = f'{files_path_on_graham}/methyl_450_pancan_tcga_reduced_10000.h5'
    mirna_file = f'{files_path_on_graham}/mirna_pancan_tcga_reduced_10000.h5'
    rna_file = f'{files_path_on_graham}/rna_pancan_tcga_reduced_10000.h5'
    # rna_iso_file = f'{files_path_on_graham}/rna_isoforms_pancan_tcga_reduced_10000.h5'
    protein_file = f'{files_path_on_graham}/protein_pancan_tcga_reduced_10000.h5'
    exon_file = f'{files_path_on_graham}/exon_pancan_tcga_reduced.h5'
    survival_file = f'{files_path_on_graham}/Survival_SupplementalTable_S1_20171025_xena_sp'
    patients_without_view_file = f'{files_path_on_graham}/patients_a_exclure_car_sans_vues.txt'
    # patients_with_one_view_file = f'{files_path_on_graham}/patients_with_one_view.txt'
    # patients_with_two_or_more_views_file = f'{files_path_on_graham}/patients_with_two_or_more_views.txt'
    # patients_with_all_4_views_available_file = f'{files_path_on_graham}/patients_with_all_4_views_available.txt'

class ReadFiles:
    def read_h5py(self, fichier: str, normalization: bool = False) -> dict:
        d = h5py.File(fichier, 'r')
        data = d['dataset'][()]
        if normalization:
            data = StandardScaler().fit_transform(data)
            # data = MinMaxScaler().fit_transform(data)
        feature_names = np.asarray([el.decode("utf-8") for el in d['features_names'][()]])
        patient_names = np.asarray([el.decode("utf-8") for el in d['patients_names'][()]])
        patient_names = dict(zip(patient_names, np.arange(len(patient_names))))
        return {'data': data, 
                'feature_names': feature_names, 
                'patient_names': patient_names}

    def read_pandas_csv(self, fichier: str):
        return pd.read_csv(fichier, sep='\t')

    def read_file_txt(self, fichier: str) -> list:
        with open(fichier, 'r') as f:
            lines = [l.strip('\n') for l in f.readlines()] 
        return lines

class BuildViews(object):
    def __init__(self, data_size: int, view_name: str):
        super(BuildViews, self).__init__()
        if data_size == 2000: pass
        if data_size == 5000:
            FichierPath.cnv_file = FichierPath5K.cnv_file
            FichierPath.methyl450_file = FichierPath5K.methyl450_file
            FichierPath.mirna_file = FichierPath5K.mirna_file
            FichierPath.rna_file = FichierPath5K.rna_file
            FichierPath.protein_file = FichierPath5K.protein_file
            # FichierPath.rna_iso_file = FichierPath5K.rna_iso_file
        if data_size == 10000:
            FichierPath.cnv_file = FichierPath10K.cnv_file
            FichierPath.methyl450_file = FichierPath10K.methyl450_file
            FichierPath.mirna_file = FichierPath10K.mirna_file
            FichierPath.rna_file = FichierPath10K.rna_file
            FichierPath.protein_file = FichierPath10K.protein_file
            # FichierPath.rna_iso_file = FichierPath10K.rna_iso_file
        if data_size not in [743, 2000, 5000, 10000]: raise ValueError(f'the data size {data_size} is not available in the dataset')
        if view_name == 'all':
            self.views = [
                ReadFiles().read_h5py(fichier=FichierPath.cnv_file, normalization=False), 
                ReadFiles().read_h5py(fichier=FichierPath.methyl450_file, normalization=False),
                ReadFiles().read_h5py(fichier=FichierPath.mirna_file, normalization=False),
                ReadFiles().read_h5py(fichier=FichierPath.rna_file, normalization=False),
                ReadFiles().read_h5py(fichier=FichierPath.protein_file, normalization=False)
            ]
        elif view_name == 'cnv':
            self.views = [
                ReadFiles().read_h5py(fichier=FichierPath.cnv_file, normalization=False)
            ]
        elif view_name == 'methyl':
            self.views = [
                ReadFiles().read_h5py(fichier=FichierPath.methyl450_file, normalization=False)
            ]
        elif view_name == 'mirna':
            self.views = [
                ReadFiles().read_h5py(fichier=FichierPath.mirna_file, normalization=False)
            ]
        elif view_name == 'rna_iso':
            self.views = [
                ReadFiles().read_h5py(fichier=FichierPath.rna_iso_file, normalization=True)
            ]
        elif view_name == 'rna':
            self.views = [
                ReadFiles().read_h5py(fichier=FichierPath.rna_file, normalization=False)
            ]
        elif view_name == 'protein':
            self.views = [
                ReadFiles().read_h5py(fichier=FichierPath.protein_file, normalization=False)
            ]
        else:
            raise ValueError(f'The view {view_name} is not available in the dataset')
        
class FilterPatientsDataset:
    def filter_patients_with_info(self, views: list, sample_to_labels: dict) -> dict:
        sample_to_labels_copy = deepcopy(sample_to_labels)
        # print('original len is', len(list(sample_to_labels.keys())))
        for name in sample_to_labels.keys():
            cpt_name = 0
            for view in views:
                if name in view['patient_names']: cpt_name += 1
            if cpt_name != 0: pass
            else: sample_to_labels_copy.pop(name)
        # print('final len is', len(list(sample_to_labels_copy.keys())))
        return sample_to_labels_copy

class MultiomicDatasetNormal(Dataset):
    def __init__(self, data_size: int = 2000, views_to_consider: str = 'all'):
        super(MultiomicDatasetNormal, self).__init__()
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
        """
        self.views = BuildViews(data_size=data_size, view_name=views_to_consider).views
        if views_to_consider == 'mirna': self.nb_features = data_size
        else: self.nb_features = np.max([view['data'].shape[1] for view in self.views])
        self.feature_names  = []
        for view in self.views:
            self.feature_names.extend(list(view['feature_names']))        
        self.survival_data = ReadFiles().read_pandas_csv(fichier=FichierPath.survival_file)
        self.sample_to_labels = {self.survival_data['sample'].values[idx]: self.survival_data['cancer type abbreviation'].values[idx] 
                                 for idx, _ in enumerate(self.survival_data['sample'].values)}
        self.sample_to_labels = FilterPatientsDataset().filter_patients_with_info(views=self.views, sample_to_labels=self.sample_to_labels)
        self.all_patient_names = np.asarray(list(self.sample_to_labels.keys()))
        self.all_patient_labels = np.asarray(list(self.sample_to_labels.values()))
        self.label_encoder = LabelEncoder() # i will need this to inverse_tranform afterward i think for the analysis downstream
        self.all_patient_labels = self.label_encoder.fit_transform(self.all_patient_labels)
        self.class_weights = compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(self.all_patient_labels),
                                                  y=self.all_patient_labels) 
        self.data_len_original = len(self.all_patient_names)

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
        original_data = data.astype(float)
        return (original_data, mask), patient_label, patient_name # i add the patient_name because we need it for an analysis downstream
    
    def __len__(self):
        return len(self.all_patient_names) 

class MultiomicDatasetDataAug(MultiomicDatasetNormal):
    def __init__(self, train_dataset: torch.utils.data.dataset.Subset, data_size: int = 2000, views_to_consider: str = 'all'):
        super().__init__(data_size=data_size, views_to_consider=views_to_consider)
        self.train_indices = train_dataset.indices 
        self.train_patient_names = train_dataset.dataset.all_patient_names[train_dataset.indices]
        for patient_name in self.all_patient_names: 
            if patient_name not in self.train_patient_names: self.sample_to_labels.pop(patient_name)
        self.all_patient_names = np.asarray(list(self.sample_to_labels.keys()))
        self.all_patient_labels = np.asarray(list(self.sample_to_labels.values()))
        self.label_encoder = LabelEncoder() # i will need this to inverse_tranform afterward i think for the analysis downstream
        self.all_patient_labels = self.label_encoder.fit_transform(self.all_patient_labels)
        self.data_len_original = len(self.all_patient_names)
        
    def __getitem__(self, idx): 
        idx = idx % self.data_len_original  # pour contrer le fait que la longueur du dataset pourrait etre supérieure à l'idx samplé
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
        # The next 2 lines are just here for debug in the future: if we have a pb with the gradient it might be due to the fact there a exempales w/o views
            # patient_name_with_matrix_vide = []
            # if np.all((data == 0)): patient_name_with_matrix_vide.append([patient_name, patient_label])
        original_mask = deepcopy(mask)
        nb_views = np.sum(mask)
        if nb_views > 1:
            # TODO: We might want or need to play here to 'turn off' a certain precise view...
            n_views_to_drop = np.random.choice(nb_views - 1)
            if n_views_to_drop >= 1:
                mask[np.random.choice(np.flatnonzero(mask), size=n_views_to_drop)] = 0
        original_data = deepcopy(data.astype(float))
        data_augmentation = data.astype(float) * mask.reshape(-1, 1) # on met à zéro la vue ou les vues qu'on a dit de drop
        return (data_augmentation, mask, original_data, original_mask), patient_label, patient_name # i add the patient_name because we need it for an analysis downstream
    
    def __len__(self):
        # Estimation de la longueur du dataset equivaut à factorial(nbre_de_vues)
        # return len(self.all_patient_names) 
        # return len(self.train_patient_names) * int(np.sqrt(math.factorial(len(self.views)))) 
        return len(self.train_patient_names) * 3
                              
class MultiomicDatasetBuilder:
    @staticmethod
    def multiomic_data_aug_builder(augmented_dataset):
        labels = [augmented_dataset[i][-1] for i in augmented_dataset.train_indices]
        # It's taking an astronomical much of time so i choose 3 to accelerate the code and see the results. 
        # It's supposed to be 10 for the next opération
        # nb_of_times_len_data_was_multiplied = int(np.sqrt(math.factorial(len(augmented_dataset.views)))) 
        nb_of_times_len_data_was_multiplied = 3
        new_labels = []
        for _ in range(nb_of_times_len_data_was_multiplied): new_labels.extend(labels)
        labels = new_labels
        new_train_dataset = Subset(augmented_dataset, indices=np.arange(len(labels)))
        return new_train_dataset
    
    @staticmethod        
    def multiomic_data_normal_builder(dataset, test_size=0.2, valid_size=0.1, random_state=42):
        n = len(dataset)
        idxs = np.arange(n)
        labels = dataset.all_patient_labels
        X_train, X_test, y_train, y_test = train_test_split(idxs, labels, test_size=test_size, random_state=random_state)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=random_state)
        train_dataset = Subset(dataset, indices=X_train)
        test_dataset = Subset(dataset, indices=X_test)
        valid_dataset =  Subset(dataset, indices=X_valid)
        return train_dataset, test_dataset, valid_dataset
    
    @staticmethod 
    def multiomic_dataset_loader(dataset, batch_size=32, nb_cpus=2):
        n = len(dataset)
        idx = np.arange(n)
        data_sampler = SubsetRandomSampler(idx)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, num_workers=nb_cpus)
        return data_loader #next(iter(test_data))[0]