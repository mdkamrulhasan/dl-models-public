#!/bin/bash

#SBATCH --gpus-per-node=1 ## nb of GPU(s)

#SBATCH --mem=8192 ##Memory I want to use in MB

#SBATCH --time=20:25:00 ## time it will take to complete job

#SBATCH --partition=class ##Partition I want to use

#SBATCH --ntasks=1 ##Number of task

#SBATCH --job-name=cnn-v1 ## Name of job

#SBATCH --output=cnn-v1-.%j.out ##Name of output file

ep=1
n_train=250

source /mnt/home/hasanka/projects/dl-models-public/dl-models-env/bin/activate

python /mnt/home/hasanka/projects/dl-models-public/cnns/cifar_cnn_pytorch.py

deactivate


