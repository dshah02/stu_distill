#!/bin/bash
#SBATCH --job-name=layer_0_kx_10_state_10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ds6237@princeton.edu

module purge
module load anaconda3/2024.6
conda activate torch-env

python train_lds.py --layer_i 0 --state_dim 10000 --batch_size 40 --epochs 8000 --seq_len 1024  --lr 0.0001 --kx 10

