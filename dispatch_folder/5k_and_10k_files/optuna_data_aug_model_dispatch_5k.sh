#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
##SBATCH --gres=gpu:v100:1
#SBATCH --mem=120G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=5:00:00

date
SECONDS=0
which python
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_version_data_augmentation.py --d_input_enc 5000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_data_aug_output_5000 --data_size 5000 --db_name experiment_data_aug_all_data_5000 --study_name experiment_all_data_5000
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date


