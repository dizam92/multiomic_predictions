import numpy as np
from collections import defaultdict
from multiomic_modeling.data.data_loader import MultiomicDataset, SubsetRandomSampler, multiomic_dataset_builder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class DatasetPlotting:
    def __init__(self, data_size: int = 2000, dataset_views_to_consider: str = 'all'):
        super(DatasetPlotting, self).__init__()
        self.dataset = MultiomicDataset(data_size=data_size, views_to_consider=dataset_views_to_consider)
    
    @staticmethod    
    def bar_plot(x: np.array, 
                 y: np.array, 
                 title: str = '', 
                 write_on_bars: bool = False,
                 rotate_xticks_labels: bool = False,
                 fig_name: str = 'plot_number_of_samples_with_n_omics', 
                 x_label: str = 'Number of omics data available per samples', 
                 y_label: str = 'Number of samples') -> None:
        fig, axes = plt.subplots(figsize=(11.69, 8.27))
        axes = sns.barplot(x=x, y=y)
        axes.set_xlabel(x_label, fontweight='bold', loc="center") # fontsize=16, 
        axes.set_ylabel(y_label, fontweight='bold', loc="center") # fontsize=16, 
        if title != '': axes.set_title(f'{title}', size=15)
        # axes.set(xlabel=x_label, ylabel=y_label)
        if rotate_xticks_labels: plt.xticks(fontsize=8, rotation=315) #-225
        if write_on_bars: 
            y_percentage = (y / sum(y)) * 100
            y_percentage = [str(np.round(el, 3)) for el in y_percentage]
            axes.bar_label(axes.containers[0], y_percentage) # if we want just the number remove the y_percentage which is the label here
        fig.savefig(f'{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{fig_name}.pdf')
        plt.close(fig)
    
    def build_fig_comptes(self) -> None:
        temporaire_dict = defaultdict(dict)
        for patient_name in self.dataset.all_patient_names:
            cpt = 0; list_of_omics_per_patients = []
            if patient_name in self.dataset.views[0]['patient_names']: cpt+=1; list_of_omics_per_patients.append('c')
            if patient_name in self.dataset.views[1]['patient_names']: cpt+=1; list_of_omics_per_patients.append('me')
            if patient_name in self.dataset.views[2]['patient_names']: cpt+=1; list_of_omics_per_patients.append('mi')
            if patient_name in self.dataset.views[3]['patient_names']: cpt+=1; list_of_omics_per_patients.append('r')
            if patient_name in self.dataset.views[4]['patient_names']: cpt+=1; list_of_omics_per_patients.append('p')
            temporaire_dict[patient_name] = [cpt, list_of_omics_per_patients]
        values_temporaire_dict = np.asarray(list(temporaire_dict.values()))
        comptes = values_temporaire_dict[:,0]
        comptes_list_of_omics_per_patients = values_temporaire_dict[:,1]
        x_comptes, y_comptes = np.unique(comptes, return_counts=True)
        x_comptes_list_of_omics_per_patients, y_comptes_list_of_omics_per_patients = np.unique(comptes_list_of_omics_per_patients, return_counts=True)
        x_comptes_list_of_omics_per_patients = np.array(['_'.join(el) for el in x_comptes_list_of_omics_per_patients])
        self.bar_plot(x=x_comptes, 
                    y=y_comptes,
                    title='', 
                    write_on_bars=True,
                    rotate_xticks_labels=False,
                    fig_name='plot_number_of_samples_with_n_omics', 
                    x_label='Number of omics data available per samples', 
                    y_label='Number of samples')
        self.bar_plot(x=x_comptes_list_of_omics_per_patients, 
                    y=y_comptes_list_of_omics_per_patients, 
                    title='',
                    write_on_bars=False,
                    rotate_xticks_labels=True,
                    fig_name='plot_number_of_samples_for_each_combination_available', 
                    x_label='Omics data combination', 
                    y_label='Number of samples')   
        cancer_names_arrays = np.asarray([self.dataset.label_encoder.inverse_transform([i])[0] for i in self.dataset.all_patient_labels])
        x_cancer_names_arrays, y_cancer_names_arrays = np.unique(cancer_names_arrays, return_counts=True)
        self.bar_plot(x=x_cancer_names_arrays, 
                      y=y_cancer_names_arrays, 
                      title='',
                      write_on_bars=False,
                      rotate_xticks_labels=True,
                      fig_name='plot_number_of_samples_per_cancer', 
                      x_label='Cancer names', 
                      y_label='Number of samples')   
    
    # def write_to_file_patient_names_with_5_views(self, saving_file: str = ''):
        #     temporaire_dict = defaultdict(dict)
        #     for patient_name in self.dataset.all_patient_names:
        #         cpt = 0; list_of_omics_per_patients = []
        #         if patient_name in self.dataset.views[0]['patient_names']: cpt+=1; list_of_omics_per_patients.append('c')
        #         if patient_name in self.dataset.views[1]['patient_names']: cpt+=1; list_of_omics_per_patients.append('me')
        #         if patient_name in self.dataset.views[2]['patient_names']: cpt+=1; list_of_omics_per_patients.append('mi')
        #         if patient_name in self.dataset.views[3]['patient_names']: cpt+=1; list_of_omics_per_patients.append('r')
        #         if patient_name in self.dataset.views[4]['patient_names']: cpt+=1; list_of_omics_per_patients.append('p')
        #         temporaire_dict[patient_name] = [cpt, list_of_omics_per_patients]
        #     values_temporaire_dict = np.asarray(list(temporaire_dict.values()))
        #     list_patients_with_five_views = [patient_name for patient_name, value in temporaire_dict.items() if value[0] == 5]
        #     with open(saving_file, 'w') as f:
        #         for name in list_patients_with_five_views: f.write(f'{name}\n')
        
        
if __name__ == '__main__':
    plot_dataset = DatasetPlotting(data_size=2000, dataset_views_to_consider='all')
    plot_dataset.build_fig_comptes()