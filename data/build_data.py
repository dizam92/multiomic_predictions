import numpy as np
import pandas as pd
import os
import pickle
import h5py
from collections import defaultdict
local_file_path_origin='/Volumes/Second Part/TCGA Pan-Cancer (PANCAN)/'
graham_file_path_origin=''
LOCAL = True
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
        chunk.index = chunk['Sample']
        patients_names = chunk.columns.values
        features_names.extend(list(chunk.index.values))
        chunk.drop('Sample', axis=1, inplace=True)
        hf.create_dataset(f'dataset_{idx}', data=chunk)
    #patients_ids = np.asarray(df_temp.index.values, dtype='str')
    features_names = [str(x).encode('utf-8') for x in features_names]
    patients_names = [str(x).encode('utf-8') for x in patients_names]
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
    print('ok')
