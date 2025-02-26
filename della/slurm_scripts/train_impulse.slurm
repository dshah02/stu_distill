#!/bin/bash
#SBATCH --job-name=impulse_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ds6237@princeton.edu

module purge
module load anaconda3/2024.6
conda activate torch-env

python lds_impulse_training.py --layer_i 2 --state_dim 2000 --seq_len 1024 --kx 10 --lr 0.0001 --epochs 5000 --output_dir results_layer2

