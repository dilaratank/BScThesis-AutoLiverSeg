#!/usr/bin/sh
#SBATCH --job-name=transfer
#SBATCH --mem=55G               # max memory per node
#SBATCH --cpus-per-task=2      # max CPU cores per MPI process, error if wrong
#SBATCH --time=00-15:00        # time limit (DD-HH:MM)
#SBATCH --gres=mps:10
#SBATCH --reservation=gpu


cd ~/scratch/AutoLiverSeg/src/
source /opt/amc/anaconda3/bin/activate

conda activate /home/dtank/scratch/autosegliver

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Starting"
nice -10 python3 transfer_learning.py
echo "Done"
