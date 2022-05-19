#!/bin/bash

n=1
#n=5
cat results/default_restore_objs/achieved_goal_end.txt | sort | uniq -c| sort -nrk 1| awk '$1> '${n} | awk '{$1=$1};1' | cut -d' ' -f2 | cut -d, -f1 > /gpfsscratch/rech/imi/usc19dv/data/UR5_processed/base_filenames_filtered.txt
