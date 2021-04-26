from multiomic_modeling.models.classification_models import TreeAndForestTemplate
from multiomic_modeling.data.data_loader import MultiomicDataset, multiomic_dataset_builder
from torch.utils.data import DataLoader
import numpy as np

def load_dataset(views_to_consider='all'):
    dataset = MultiomicDataset(views_to_consider=views_to_consider, type_of_model='mlp')
    train_dataset, test_dataset, valid_dataset = multiomic_dataset_builder(dataset, test_size=0.2, valid_size=0.1)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    train_dataset_array = next(iter(train_loader))[0].numpy()
    train_dataset_array_labels = next(iter(train_loader))[1].numpy()
    
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    x_test = next(iter(test_loader))[0].numpy()
    y_test = next(iter(test_loader))[1].numpy()
    
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    valid_dataset_array = next(iter(valid_loader))[0].numpy()
    valid_dataset_array_labels = next(iter(valid_loader))[1].numpy()
    
    x_train = np.vstack((train_dataset_array, valid_dataset_array))
    y_train = np.hstack((train_dataset_array_labels, valid_dataset_array_labels))
    
    feature_names = train_dataset.dataset.feature_names
    
    return x_train, y_train, x_test, y_test, feature_names

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, feature_names = load_dataset(views_to_consider='all')
    dt_clf = TreeAndForestTemplate(algo='tree')
    rf_clf = TreeAndForestTemplate(algo='rf')
    dt_clf.learn(x_train=x_train, y_train=y_train, 
                 x_test=x_test, y_test=y_test, 
                 feature_names=feature_names, saving_file='/home/maoss2/scratch/expts/decison_trees_scores.json')
    rf_clf.learn(x_train=x_train, y_train=y_train, 
                 x_test=x_test, y_test=y_test, 
                 feature_names=feature_names, saving_file='/home/maoss2/scratch/expts/rf_trees_scores.json')