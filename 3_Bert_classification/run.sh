#!/bin/bash
#SBATCH --output=records.log
#SBATCH --error=records.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gputest
#SBATCH --account=project_2006362

module load pytorch/2.2

source /scratch/project_2003238/v/envs/BERT/bin/activate

python3 run.py --model bert