#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=10:00:00

date
SECONDS=0
which python
# python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_dnn.py --input_size 10000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_dnn_all_output_3_layers --data_size 2000 --db_name experiment_dnn_all_data_2000 --study_name experiment_all_data_2000
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_dnn.py --input_size 2000 --dataset_views_to_consider cnv --output_path /home/maoss2/scratch/optuna_dnn_cnv_output_3_layers --data_size 2000 --db_name experiment_dnn_cnv_data_2000 --study_name experiment_all_data_2000
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_dnn.py --input_size 2000 --dataset_views_to_consider methyl --output_path /home/maoss2/scratch/optuna_dnn_methyl_output_3_layers --data_size 2000 --db_name experiment_dnn_methyl_data_2000 --study_name experiment_all_data_2000
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_dnn.py --input_size 2000 --dataset_views_to_consider mirna --output_path /home/maoss2/scratch/optuna_dnn_mirna_output_3_layers --data_size 2000 --db_name experiment_dnn_mirna_data_2000 --study_name experiment_all_data_2000
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_dnn.py --input_size 2000 --dataset_views_to_consider rna --output_path /home/maoss2/scratch/optuna_dnn_rna_output_3_layers --data_size 2000 --db_name experiment_dnn_all_rna_2000 --study_name experiment_all_data_2000
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_dnn.py --input_size 2000 --dataset_views_to_consider protein --output_path /home/maoss2/scratch/optuna_dnn_protein_output_3_layers --data_size 2000 --db_name experiment_dnn_protein_data_2000 --study_name experiment_all_data_2000

diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date


