#!/bin/bash
#SBATCH --job-name=eval_demos_tw
#SBATCH -A imi@cpu
##SBATCH --qos=qos_gpu-dev
##SBATCH --partition=gpu_p2
#SBATCH --ntasks=32
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
##SBATCH --output=out.out
##SBATCH --error=err.err


export ROOT_FOLDER=/gpfswork/rech/imi/usc19dv/captionRLenv/
export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5/
export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/


srun -n 32 python3 evaluate_mpi.py --eval_train_demos --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames.txt

