#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=3:00:00

date
SECONDS=0
which python
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/models_optuna_dnn.py --input_size 10000 --dataset_views_to_consider all --output_path /home/maoss2/scratch/optuna_dnn_output_3_layers --data_size 2000 --db_name experiment_dnn_all_data_2000 --study_name experiment_all_data_2000
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date


