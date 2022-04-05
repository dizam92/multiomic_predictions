# multiomic_predictions
This is a the implementation of the model (MOT) submited at BMC. The preprint is available here: [https://doi.org/10.21203/rs.3.rs-1348696/v1].
MOT (Multi-Omic Transformer) is a deep learning based model using the transformer architecture, that discriminates complex phenotypes (herein cancers types) based on five omics data type regardless of their availability: transcriptomics (mRNA and miRNA), epigenomics (DNA methylation), copy number variations (CNVs), and proteomics. 
The Pancan Dataset is available at [https://xenabrowser.net/datapages/?hub=https://pancanatlas.xenahubs.net:443]. 

## STEPS
# Building Dataset
1. Download the files from The Pancan Dataset at [https://xenabrowser.net/datapages/?hub=https://pancanatlas.xenahubs.net:443]. 
2. Run [https://github.com/dizam92/multiomic_predictions/blob/76430ac1719f478f8799c1df8e4e4f1a064036ed/multiomic_modeling/data/build_data.py#L117] (Change the path for the dataset)

# Models
1. The normal version (without data augmentation): [https://github.com/dizam92/multiomic_predictions/blob/main/multiomic_modeling/models/models_optuna_version_normal.py]

2. The version (with data augmentation): [https://github.com/dizam92/multiomic_predictions/blob/main/multiomic_modeling/models/models_optuna_version_data_augmentation.py]

# Analysis section
[https://github.com/dizam92/multiomic_predictions/tree/main/multiomic_modeling/analyses_and_paper_figures_section]
