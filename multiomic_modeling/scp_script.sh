#!/bin/bash

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/previous_results/optuna_test_output_2000/0d5978314e7252b03bbd8515afa42b7c8c1621fc/' ./
mv 0d5978314e7252b03bbd8515afa42b7c8c1621fc normal_output_2000

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/previous_results/optuna_test_output_5000/7ba5aef75ab91cd2d985129435911522df496730/' ./
mv 7ba5aef75ab91cd2d985129435911522df496730 normal_output_5000

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/previous_results/optuna_test_output_10000/5e5bccac73c3be92f11dd69011a15f2a4df03ca4/' ./
mv 5e5bccac73c3be92f11dd69011a15f2a4df03ca4 normal_output_10000

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/optuna_test_output_2000/f3af0f144fd1e2afee48276156d57d3f85703384/' ./
mv f3af0f144fd1e2afee48276156d57d3f85703384 data_augmentation_output_2000

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/optuna_test_output_5000/cbd211752368f75fc6cf2984bd9dd3474c25929a/' ./
mv cbd211752368f75fc6cf2984bd9dd3474c25929a data_augmentation_output_5000

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/optuna_test_output_10000/2a8d6486786b0776edcad273876d96e22968ead3/' ./
mv 2a8d6486786b0776edcad273876d96e22968ead3 data_augmentation_output_10000

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/optuna_multimodal_test_output_2000/46ab0c165431876b737d36804bb926e834801517/' ./
mv 46ab0c165431876b737d36804bb926e834801517 data_augmentation_multimodal_output_2000

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/optuna_multimodal_test_output_5000/2bbf7f409f49bc0abacb7838cd5516dcc52d65f0/' ./
mv 2bbf7f409f49bc0abacb7838cd5516dcc52d65f0 data_augmentation_multimodal_output_5000

scp -r maoss2@graham.computecanada.ca:'/home/maoss2/scratch/optuna_multimodal_test_output_10000/76c336aee9bdeee6ca988f31124e75b21a8f0715/' ./
mv 76c336aee9bdeee6ca988f31124e75b21a8f0715 data_augmentation_multimodal_output_10000

