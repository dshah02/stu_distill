#!/bin/bash
#SBATCH --job-name=training_lds_stu_pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=0:01:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ds6237@princeton.edu

module purge
module load anaconda3/2024.6
conda activate torch-env

python train_lds_stu_pairs.py --prefix 0 --steps 5000 --num_models 5000

