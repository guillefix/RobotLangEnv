#!/bin/bash
#SBATCH --job-name=resample_demos_tw
#SBATCH -A imi@cpu
##SBATCH --qos=qos_gpu-dev
##SBATCH --partition=gpu_p2
#SBATCH --ntasks=320
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
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


srun --wait=0 -n 320 python3 inference_mpi.py --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames.txt --save_relabelled_trajs --restore_objects --experiment_name default_restore_objs_resample --num_repeats 5
#srun -n 320 python3 inference_mpi.py --using_model --experiment_name train_transflower_zp5_single_obj_nocol_trim_tw_single_filtered_restore_objs --pretrained_name transflower_zp5_single_obj_nocol_trim_tw_single_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered.txt --save_eval_results --save_sampled_traj --num_repeats 20 --restore_objects
