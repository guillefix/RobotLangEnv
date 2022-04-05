import numpy as np
import os
import scipy.ndimage.filters as filters

from data_utils import fix_quaternions

def process_file(filePath, save_folder, suffix="", smoothing=0):
    seq_id = "_".join(filePath.split("/"))
    a = np.load(filePath)
    acts = a['acts']
    obs = a['obs']
    #fix quaternions
    acts[:,3:7] = fix_quaternions(acts[:,3:7])
    obs[:,3:7] = fix_quaternions(obs[:,3:7])

    if smoothing > 0:
        acts = filters.uniform_filter1d(acts, smoothing, mode='nearest', axis=0)

    np.save(save_folder+"/"+seq_id+".acts"+suffix, acts)
    np.save(save_folder+"/"+seq_id+".obs"+suffix, obs)
    open(save_folder+"/"+seq_id+".acts.npy.annotation.txt"+suffix, "w").write(a['goal_str'][0])


# filePath="UR5/Laetitia/obs_act_etc/66/data.npz"
# seq_id = "_".join(filePath.split("/"))
# a = np.load(filePath)
# acts = a['acts']
# obs = a['obs']
# # obs.shape
# obs[3]
#fix quaternions
# acts[3:7] = fix_quaternions(acts[3:7])
# obs[3:7] = fix_quaternions(obs[3:7])
#
# np.load("UR5_processed/"+seq_id+".obs.npy")[3]

root_folder="UR5"
save_folder="UR5_processed"
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
for dirpath, dirs, files in os.walk(root_folder):
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if fname.endswith('.npz'):
            print(fname)
            process_file(fname, save_folder, smoothing=5)
