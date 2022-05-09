#!/bin/bash
#source ~/.bashrc
export sb=/linkhome/rech/genini01/usc19dv/sbatch.sh
#export job0=$($sb run_scripts/resample_data.slurm)
#echo $job0
#export job1=$($sb --dependency=afterok:$job0 run_scripts/process_data.slurm)
export job1=$($sb run_scripts/process_data.slurm)
echo $job1
export job2=$($sb --dependency=afterok:$job1 run_scripts/list_base_filenames.slurm)
echo $job2
export job3=$($sb --dependency=afterok:$job2 run_scripts/create_simple_dataset.slurm)
echo $job3
export job4=$($sb --dependency=afterok:$job3 run_scripts/extract_features.slurm)
echo $job4
