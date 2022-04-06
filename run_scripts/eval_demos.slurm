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


module load pytorch-gpu/py3/1.10.0

export ROOT_FOLDER=/gpfswork/rech/imi/usc19dv/captionRLenv/
export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5/
export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/
export PRETRAINED_FOLDER=/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/


srun -n 32 python3 inference_mpi.py --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames.txt --save_eval_results
#srun -n 32 python3 inference_mpi.py --using_model --experiment_name train_transflower_zp5_single_obj_nocol_trim_tw_single_filtered --pretrained_name transflower_zp5_single_obj_nocol_trim_tw_single_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames.txt --save_eval_results
