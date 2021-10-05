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
python3 /home/maoss2/PycharmProjects/multiomic_predictions/multiomic_modeling/models/test_with_new_weights.py
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date


