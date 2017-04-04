#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=60GB
#SBATCH --job-name=vnmt
#SBATCH --mail-type=END
#SBATCH --mail-user=jgt275@nyu.edu
#SBATCH --output=slurm_%j.out
 
module purge
  
SRCDIR=$HOME/VNMT
RUNDIR=$SCRATCH/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
mkdir -p $RUNDIR/runs
cd $RUNDIR
  
module load pytorch/intel/20170226
module load tensorboard_logger/0.0.3

python $SRCDIR/train.py -data $SRCDIR/data/200k_LM.pt-train.pt -save_model model -logdir runs/ -brnn -gpu 0 -batch_size 72 -sample 5


