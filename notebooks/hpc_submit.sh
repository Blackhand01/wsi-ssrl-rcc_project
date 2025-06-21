#!/bin/bash
#SBATCH --job-name=rcc_ssrl_launch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=global
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s333962@studenti.polito.it
#SBATCH --workdir=/home/mla_group_19/wsi-ssrl-rcc_project

module purge
module load python/3.9

cd /home/mla_group_19/wsi-ssrl-rcc_project

python /Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project/4-launch_training.py --config config/training.yaml
