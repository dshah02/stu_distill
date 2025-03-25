#!/bin/bash
#SBATCH --job-name=training_lds_stu_pairs_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ds6237@princeton.edu
#SBATCH --constraint=cascade

module purge
module load anaconda3/2024.6
conda activate torch-env

python train_lds_stu_pairs.py --prefix cpu_1_ --steps 5000 --num_models 5000

