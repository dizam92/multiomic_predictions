#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=40:00:00

date
SECONDS=0
which python
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 10000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_data_aug_output_10000 --data_size 10000 --db_name experiment_data_aug_all_data_10000 --study_name experiment_all_data_10000
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date


