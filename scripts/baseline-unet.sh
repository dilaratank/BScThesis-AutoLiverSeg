#!/bin/sh
#SBATCH --job-name=unet
#SBATCH --mem=55G               # max memory per node
#SBATCH --cpus-per-task=2      # max CPU cores per MPI process, error if wrong
#SBATCH --time=00-15:00        # time limit (DD-HH:MM)
###SBATCH --gres=mps:8   # dit script heeft een volledige GPU nodig, dus gebruik geen mps
#SBATCH --gres=gpu:p100:1      # number of p100 GPUs: dus 1 volledige P100 GPU
#SBATCH --reservation=gpu


cd ~/scratch/AutoLiverSeg/src/
source /opt/amc/anaconda3/bin/activate

conda activate /home/dtank/scratch/autosegliver

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Starting"
nice -10 python3 train.py --arch 'unet' --name 'unet'
echo "Done"
