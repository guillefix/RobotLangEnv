import numpy as np
import os
import scipy.ndimage.filters as filters
from extra_utils.data_utils import fix_quaternions, get_obs_cont, get_ann_with_obj_types, get_tokens
from constants import *

def process_file(filePath, save_folder, mods=["obs", "acts"], smoothing=0):
    seq_id = "_".join(filePath.split("/"))
    a = np.load(filePath)
    acts = a['acts']
    obs = a['obs']
    if len(acts) > 0 and len(obs) > 0:
        #fix quaternions
        # print(filePath)
        # print(acts.shape)
        acts[:,3:7] = fix_quaternions(acts[:,3:7])
        obs[:,3:7] = fix_quaternions(obs[:,3:7])
        ann = a['goal_str'][0]
        open(save_folder+"/"+seq_id+".annotation.txt", "w").write(ann)

        for mod in mods:
            if mod == "obs":
                np.save(save_folder+"/"+seq_id+".obs", obs)
            if mod == "acts":
                if smoothing > 0:
                    acts = filters.uniform_filter1d(acts, smoothing, mode='nearest', axis=0)
                np.save(save_folder+"/"+seq_id+".acts", acts)
            if mod == "obs_cont":
                obs_cont = get_obs_cont(obs)
                np.save(save_folder+"/"+seq_id+".obs_cont", obs_cont)
            if mod == "disc_cond":
                tokens = get_tokens(ann)[:,0]
                disc_cond = get_ann_with_obj_types(tokens, obs)
                np.save(save_folder+"/"+seq_id+".disc_cond", disc_cond)
    else:
        print("ZERO-LENGTH SEQUENCE: "+filePath)


from extra_utils import distribute_tasks
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)
dirs = data_folder.split("/")
os.chdir(data_folder+"..")
tasks = list(os.walk(dirs[-2]))
tasks = distribute_tasks(tasks, rank, size)

if not os.path.isdir(processed_data_folder):
    os.mkdir(processed_data_folder)
for dirpath, dirs, files in tasks:
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if fname.endswith('.npz'):
            print(fname)
            # process_file(fname, processed_data_folder, mods=["obs", "acts"], smoothing=5)
            process_file(fname, processed_data_folder, mods=["obs_cont", "disc_cond"], smoothing=5)
