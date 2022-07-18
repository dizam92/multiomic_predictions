#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=10:00:00

# random_seed_list = [42, 78, 433, 966, 699]
date
SECONDS=0
which python
# Prepare virtualenv
source ~/jupyter_py3/bin/activate
which python

# python /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider 3_main_omics --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_3_main_omics_42 --data_size 2000 --db_name experiment_data_aug_3_main_omics_data_2000 --study_name experiment_3_main_omics_data_2000 --seed 42

# python /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider 3_main_omics --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_3_main_omics_78 --data_size 2000 --db_name experiment_data_aug_3_main_omics_data_2000 --study_name experiment_3_main_omics_data_2000 --seed 78

python /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider 3_main_omics --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_3_main_omics_433 --data_size 2000 --db_name experiment_data_aug_3_main_omics_data_2000 --study_name experiment_3_main_omics_data_2000 --seed 433

# python /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider 3_main_omics --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_3_main_omics_966 --data_size 2000 --db_name experiment_data_aug_3_main_omics_data_2000 --study_name experiment_3_main_omics_data_2000 --seed 966

# python /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider 3_main_omics --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_3_main_omics_699 --data_size 2000 --db_name experiment_data_aug_3_main_omics_data_2000 --study_name experiment_3_main_omics_data_2000 --seed 699


diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date


