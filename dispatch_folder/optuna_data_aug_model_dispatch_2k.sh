#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=14:00:00

# random_seed_list = [42, 78, 433, 966, 699]
date
SECONDS=0
which python
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_42 --data_size 2000 --db_name experiment_data_aug_all_data_2000 --study_name experiment_all_data_2000 --seed 42
# python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_78 --data_size 2000 --db_name experiment_data_aug_all_data_2000 --study_name experiment_all_data_2000 --seed 78
# python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_433 --data_size 2000 --db_name experiment_data_aug_all_data_2000 --study_name experiment_all_data_2000 --seed 433
# python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_966 --data_size 2000 --db_name experiment_data_aug_all_data_2000 --study_name experiment_all_data_2000 --seed 966
# python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 2000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_data_aug_output_2000_699 --data_size 2000 --db_name experiment_data_aug_all_data_2000 --study_name experiment_all_data_2000 --seed 699
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date


