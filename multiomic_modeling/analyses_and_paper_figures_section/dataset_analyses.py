import numpy as np
from collections import defaultdict
from multiomic_modeling.data.data_loader import MultiomicDataset, SubsetRandomSampler, multiomic_dataset_builder

def comptes():
    data_size = 2000; dataset_views_to_consider = 'all'
    dataset = MultiomicDataset(data_size=data_size, views_to_consider=dataset_views_to_consider)
    temporaire_dict = defaultdict(dict)
    cpt_cnv = 0; cpt_mirna = 0; cpt_rna = 0; cpt_methyl = 0
    for patient_name in dataset.all_patient_names:
        cpt = 0
        if patient_name in dataset.views[0]['patient_names']: cpt_cnv+=1; cpt+=1
        if patient_name in dataset.views[1]['patient_names']: cpt_methyl+=1; cpt+=1
        if patient_name in dataset.views[2]['patient_names']: cpt_mirna+=1; cpt+=1
        if patient_name in dataset.views[3]['patient_names']: cpt_rna+=1; cpt+=1
        temporaire_dict[patient_name] = cpt
    temp_array = np.asarray(list(temporaire_dict.values()))
    print(np.unique(temp_array))
    for el in np.unique(temp_array):
        print(np.where(temp_array==el)[0].shape)