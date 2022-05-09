#!/bin/bash
#SBATCH --job-name=list_base_filenames
##SBATCH --job-name=resample_demos_tw
#SBATCH -A imi@cpu
##SBATCH --qos=qos_gpu-dev
##SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
##SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
##SBATCH --qos=qos_cpu-dev
#SBATCH --qos=qos_cpu-t3 
##SBATCH --output=out.out
##SBATCH --error=err.err


module purge
#module load pytorch-gpu/py3/1.10.0
module load pytorch-cpu/py3/1.7.1

export ROOT_FOLDER=/gpfswork/rech/imi/usc19dv/captionRLenv/
export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5/
export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/
export PRETRAINED_FOLDER=/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/


srun -n 1 ../mt-lightning/feature_extraction/script_to_list_filenames /gpfsscratch/rech/imi/usc19dv/data/generated_data_processed/ npz.disc_cond.npy
