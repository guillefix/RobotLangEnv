#!/bin/bash
##SBATCH --job-name=eval_demos_tw
#SBATCH --job-name=eval_model_tw_gen
#SBATCH -A imi@cpu
#SBATCH --qos=qos_cpu-dev
##SBATCH --qos=qos_gpu-dev
##SBATCH --partition=gpu_p2
##SBATCH --ntasks=4
#SBATCH --ntasks=80
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
##SBATCH --time=2:00:00
##SBATCH --qos=qos_cpu-dev
#SBATCH --qos=qos_cpu-t3 
##SBATCH --output=out.out
##SBATCH --error=err.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guillefix@gmail.com


module purge
#module load pytorch-gpu/py3/1.10.0
module load pytorch-cpu/py3/1.7.1

export ROOT_FOLDER=/gpfswork/rech/imi/usc19dv/captionRLenv/
export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5/
export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/
export PRETRAINED_FOLDER=/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/
export ROOT_GENERATED_DATA=/gpfsscratch/rech/imi/usc19dv/data/

#srun -n 32 python3 inference_mpi.py --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames.txt --save_eval_results
srun --wait=0 -n 80 python3 inference_mpi.py --using_model --experiment_name transflower_zp5_short4_tw_filtered_norestore_objs --pretrained_name transflower_zp5_short4_tw_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_filtered.txt --save_eval_results --save_relabelled_trajs --num_repeats 20
#srun --wait=0 -n 320 python3 inference_mpi.py --using_model --experiment_name transflower_zp5_short_single_obj_nocol_trim_tw_single_filtered_norestore_objs --pretrained_name transflower_zp5_short_single_obj_nocol_trim_tw_single_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered.txt --save_eval_results --save_relabelled_trajs --num_repeats 20
#srun --wait=0 -n 24 python3 inference_mpi.py --using_model --experiment_name testing_transflower_zp5_short_single_obj_nocol_trim_tw_single_filtered_norestore_objs --pretrained_name transflower_zp5_short_single_obj_nocol_trim_tw_single_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered_testing.txt --save_eval_results --save_relabelled_trajs --num_repeats 20
#srun --wait=0 -n 320 python3 inference_mpi.py --using_model --experiment_name transflower_zp5_long_single_obj_nocol_trim_tw_single_filtered_norestore_objs --pretrained_name transflower_zp5_long_single_obj_nocol_trim_tw_single_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered.txt --save_eval_results --save_relabelled_trajs --num_repeats 20
#srun --wait=0 -n 320 python3 inference_mpi.py --using_model --experiment_name transflower_zp5_long2_single_obj_nocol_trim_tw_single_filtered_norestore_objs --pretrained_name transflower_zp5_long2_single_obj_nocol_trim_tw_single_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered.txt --save_eval_results --save_relabelled_trajs --num_repeats 20
#srun --wait=0 -n 320 python3 inference_mpi.py --using_model --experiment_name transflower_zp5_long2_single_obj_nocol_trim_tw_single_more_filtered_norestore_objs --pretrained_name transflower_zp5_long2_single_obj_nocol_trim_tw_single_more_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered.txt --save_eval_results --save_relabelled_trajs --num_repeats 20
#srun --wait=0 -n 320 python3 inference_mpi.py --using_model --experiment_name transflower_zp5_long_single_obj_nocol_trim_tw_single_more_filtered_norestore_objs --pretrained_name transflower_zp5_long_single_obj_nocol_trim_tw_single_more_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered.txt --save_eval_results --save_relabelled_trajs --num_repeats 20
#srun --wait=0 -n 320 python3 inference_mpi.py --using_model --experiment_name transflower_zp5_short_single_obj_nocol_trim_tw_single_more_filtered_norestore_objs --pretrained_name transflower_zp5_short_single_obj_nocol_trim_tw_single_more_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered.txt --save_eval_results --save_relabelled_trajs --num_repeats 20

#export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/generated_data/
#export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/generated_data_processed/
#srun -n 320 python3 inference_mpi.py --using_model --experiment_name transflower_udrl_short_single_obj_nocol_trim_tw_udrl_single_filtered_norestore_objs --pretrained_name transflower_udrl_short_single_obj_nocol_trim_tw_udrl_single_filtered --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs.txt --save_eval_results --save_relabelled_trajs --num_repeats 20
