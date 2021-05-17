import numpy as np
import pandas as pd
import os
import pickle
import h5py
from collections import defaultdict
from scipy.stats import median_abs_deviation
from sklearn.feature_selection import SelectKBest, mutual_info_classif
local_file_path_origin='/Volumes/Second Part/TCGA Pan-Cancer (PANCAN)/'
graham_file_path_origin='/home/maoss2/project/maoss2/tcga_pan_cancer_dataset'
LOCAL = False
def read_chunk_file(fichier_path, saving_file_name, chunk_size=100000):
    """
    Read the CSV file with th chunk_size to fit in memory.
    Build a hdf5? pickle? file from it?*** to be determined
    """
    hf = h5py.File(f'{saving_file_name}', 'w')
    fichier_read_chunk = pd.read_csv(fichier_path, sep='\t', chunksize=chunk_size)
    features_names = []
    patients_names = []
    for idx, chunk in enumerate(fichier_read_chunk):
        try:
            chunk.index = chunk['Sample']
            chunk.drop('Sample', axis=1, inplace=True)
        except KeyError:
            chunk.index = chunk['sample'] 
            chunk.drop('sample', axis=1, inplace=True)
        patients_names = chunk.columns.values
        features_names.extend(list(chunk.index.values))
        hf.create_dataset(f'dataset_{idx}', data=chunk)
    features_names = [str(x).encode('utf-8') for x in features_names]
    patients_names = [str(x).encode('utf-8') for x in patients_names]
    hf.create_dataset('features_names', data=features_names)
    hf.create_dataset('patients_names', data=patients_names)
    hf.close()

def select_features_based_on_mad(x, axe=0, nb_features=5000):
    """
    Utility function to help build the mad. Compute the mad for each features
    and make a sort on the features to take the n best features
    Args:
        x, numpy array, data of each view
        axe, int, 0 or 1: if 0 run on the columns, if 1 run on the row (Unconventional cause i'm using a stats library)
        nb_features, int, default number of feature to be selected
    Return:
        indices_features, the indices in the array of the features to be selected
    """
    assert axe in [0, 1], f'Can not do on axe {axe}'
    mad_all_features = median_abs_deviation(x, axis=axe) #, scale='normal'
    indices_features = np.argsort(mad_all_features)[::-1]
    return indices_features[:nb_features]

def build_file_with_dimentionality_reduction(fichier_path, saving_file_name, nb_features_selected=2000):
    hf = h5py.File(f'{saving_file_name}', 'w')
    data = pd.read_csv(fichier_path, sep='\t')
    if 'Sample' in data.columns.values:
        data.index = data['Sample']
        data.drop('Sample', axis=1, inplace=True)
    if 'sample' in data.columns.values:
        data.index = data['sample']
        data.drop('sample', axis=1, inplace=True)
    if 'SampleID' in data.columns.values:
        data.index = data['SampleID']
        data.drop('SampleID', axis=1, inplace=True) 
    data.dropna(axis=0, inplace=True)
    patients_names = data.columns.values
    features_names = data.index.values
    data = data.values
    if fichier_path.find('cnv') != -1:
        y = np.ones(data.shape[0])
        learner = SelectKBest(mutual_info_classif, k=nb_features_selected).fit(X=data, y=y)
        indices_selected = learner.get_support(indices=True)
        data = learner.transform(data)
        features_names = features_names[indices_selected]
        del y, learner, indices_selected
    else:
        indices_mad_selected = select_features_based_on_mad(x=data, axe=1, nb_features=nb_features_selected)
        data = data[indices_mad_selected]
        features_names = features_names[indices_mad_selected]
        del indices_mad_selected
    features_names = [str(x).encode('utf-8') for x in features_names]
    patients_names = [str(x).encode('utf-8') for x in patients_names]
    hf.create_dataset(f'dataset', data=data.T)
    hf.create_dataset('features_names', data=features_names)
    hf.create_dataset('patients_names', data=patients_names)
    hf.close()
        
if LOCAL:
    exon_path = f'{local_file_path_origin}/HiSeqV2_exon'
    cnv_path = f'{local_file_path_origin}/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes'
    methyl_27_path = f'{local_file_path_origin}/HumanMethylation27'
    methyl_450_path = f'{local_file_path_origin}/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena'
    rna_path = f'{local_file_path_origin}/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena'
    rna_isoforms_path = f'{local_file_path_origin}/tcga_RSEM_isoform_fpkm'
    mirna_path = f'{local_file_path_origin}/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena'
    protein_path = f'{local_file_path_origin}/TCGA-RPPA-pancan-clean.xena'
else:
    exon_path = f'{graham_file_path_origin}/HiSeqV2_exon'
    cnv_path = f'{graham_file_path_origin}/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes'
    methyl_27_path = f'{graham_file_path_origin}/HumanMethylation27'
    methyl_450_path = f'{graham_file_path_origin}/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena'
    rna_path = f'{graham_file_path_origin}/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena'
    rna_isoforms_path = f'{graham_file_path_origin}/tcga_RSEM_isoform_fpkm'
    mirna_path = f'{graham_file_path_origin}/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena'
    protein_path = f'{graham_file_path_origin}/TCGA-RPPA-pancan-clean.xena'


if __name__ == '__main__':
    # TODO: Si la liste fichiers_path change, il faut update la liste saving_files_names car chaque nom a été écrit pour le fichier correspondant
    fichiers_path = [cnv_path, methyl_450_path, rna_path, mirna_path]
    # saving_files_names = ['exon_pancan_tcga.h5', 'cnv_pancan_tcga.h5', 'methyl_27_pancan_tcga.h5', 'methyl_450_pancan_tcga.h5', 
    #                       'rna_pancan_tcga.h5', 'rna_isoforms_pancan_tcga.h5', 'mirna_pancan_tcga.h5', 'protein_pancan_tcga.h5']
    # saving_files_names_reduced = ['exon_pancan_tcga_reduced', 'cnv_pancan_tcga_reduced', 'methyl_27_pancan_tcga_reduced', 
    #                               'methyl_450_pancan_tcga_reduced', 'rna_pancan_tcga_reduced', 'rna_isoforms_pancan_tcga_reduced', 
    #                               'mirna_pancan_tcga_reduced', 'protein_pancan_tcga_reduced']
    saving_files_names = ['cnv_pancan_tcga.h5', 'methyl_450_pancan_tcga.h5', 'rna_pancan_tcga.h5','mirna_pancan_tcga.h5']
    saving_files_names_reduced = ['cnv_pancan_tcga_reduced', 'methyl_450_pancan_tcga_reduced', 'rna_pancan_tcga_reduced', 'mirna_pancan_tcga_reduced']
    for idx, fichier in enumerate(fichiers_path):
        # read_chunk_file(fichier_path=fichier, saving_file_name=f'{graham_file_path_origin}/data_hdf5/{saving_files_names[idx]}', chunk_size=100000)
        build_file_with_dimentionality_reduction(fichier_path=fichier, 
                                                 saving_file_name=f'{graham_file_path_origin}/data_hdf5/{saving_files_names_reduced[idx]}_2000.h5',
                                                 nb_features_selected=2000)
        build_file_with_dimentionality_reduction(fichier_path=fichier, 
                                                 saving_file_name=f'{graham_file_path_origin}/data_hdf5/{saving_files_names_reduced[idx]}_5000.h5',
                                                 nb_features_selected=5000)
        build_file_with_dimentionality_reduction(fichier_path=fichier, 
                                                 saving_file_name=f'{graham_file_path_origin}/data_hdf5/{saving_files_names_reduced[idx]}_10000.h5',
                                                 nb_features_selected=10000)

