#!/bin/bash
##SBATCH --nodes=1
#SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:v100:1
#SBATCH --mem=2G
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=mazid-abiodoun.osseni.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=00:01:00

date
SECONDS=0
which python
# Prepare virtualenv
source ~/jupyter_py3/bin/activate
which python
