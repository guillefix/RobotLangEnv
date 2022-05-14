import numpy as np
import glob
fs1 = [x[:-1] for x in open("/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/base_filenames.txt", "r").readlines()]
fs2 = ["results/default_restore_objs4/"+f.split("_")[1]+"_"+f.split("_")[5]+"_*.txt" for f in fs1]
fs3 = [glob.glob(f)[0] for f in fs2]
fs4 = [np.arange(int(open(f, "r").readlines()[1][:-1].split(",")[2]), 0, -1) for f in fs3]
fs_ttg = [np.zeros_like(np.load("/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/"+f+".npz.times_to_go.npy")) for f in fs1]
for i in range(len(fs_ttg)): fs_ttg[i][:fs4[i].shape[0]] = fs4[i][:,None]
for i in range(len(fs_ttg)): np.save("/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/"+fs1[i]+".npz.times_to_go.npy",fs_ttg[i])

