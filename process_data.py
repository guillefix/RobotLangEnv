import torch
import os
print("OWO2")
from extra_utils import distribute_tasks
print("OWO3")
from mpi4py import MPI
print("OWO1")
comm = MPI.COMM_WORLD
print("OWO0")
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)
import numpy as np
import scipy.ndimage.filters as filters
from extra_utils.data_utils import fix_quaternions, get_obs_cont, get_ann_with_obj_types, get_tokens
from constants import *
import json
try:
    vocab_file=json.load(open(processed_data_folder+"npz.annotation.txt.annotation.class_index.json", "r"))
except Exception as e:
    print(e)
    vocab_file = None
    print("no vocab file yet at "+processed_data_folder)

import argparse
parser = argparse.ArgumentParser(description='Process data')
parser.add_argument('--data_folder', default=None, help='folder from which to read data')
parser.add_argument('--processed_data_folder', default=None, help='folder to which to write data')

def process_file(filePath, save_folder, mods=["obs", "acts"], smoothing=0):
    seq_id = "_".join(filePath.split("/"))
    a = np.load(filePath)
    acts = a['acts']
    obs = a['obs']
    # print(obs)
    # print(acts)
    if len(acts) > 0 and len(obs) > 0:
        #fix quaternions
        # print(filePath)
        # print(acts.shape)
        acts[:,3:7] = fix_quaternions(acts[:,3:7])
        obs[:,3:7] = fix_quaternions(obs[:,3:7])
        with open(save_folder+"/"+seq_id+".annotation.txt", "w") as f:
            for ann in a['goal_str']:
                # ann = a['goal_str'][0]
                f.write(ann+"\n")
        if vocab_file is not None:
            for i,ann in enumerate(a['goal_str']):
                tokens = get_tokens(ann)[None]
                if i == 0:
                    tokenss = tokens
                else:
                    tokenss = np.concatenate([tokenss,tokens])
            np.save(save_folder+"/"+seq_id+".annotation.txt.annotation", tokenss)
        L = obs.shape[0]
        times_to_go = np.expand_dims(np.array(range(L)),1)
        np.save(save_folder+"/"+seq_id+".times_to_go", times_to_go)

        # actually one of the elements is constant, because the pad colour has four elements for some reason and the fourth element is constant... (remember see instance.py to see the menaing of each of the obs dimensions)
        # maybe should remove the constant element..
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
                for ii,ann in enumerate(a['goal_str']):
                    tokens = get_tokens(ann)[:,0]
                    disc_cond = np.expand_dims(get_ann_with_obj_types(tokens, obs),1)[None]
                    if ii == 0:
                        disc_conds = disc_cond
                    else:
                        disc_conds = np.concatenate([disc_conds,disc_cond])
                np.save(save_folder+"/"+seq_id+".disc_cond", disc_conds)
    else:
        print("ZERO-LENGTH SEQUENCE: "+filePath)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.data_folder is not None:
        if args.data_folder[0] != "/":
            data_folder = os. getcwd() + "/" + args.data_folder
        else:
            data_folder = args.data_folder
    if args.processed_data_folder is not None:
        if args.processed_data_folder[0] != "/":
            processed_data_folder = os. getcwd() + "/" + args.processed_data_folder
        else:
            processed_data_folder = args.processed_data_folder
    dirs = data_folder.split("/")
    if len(dirs[-1]) == 0:
        dirs = dirs[:-1]
    os.chdir(data_folder+"/..")
    tasks = list(os.walk(dirs[-1]))
    #print(tasks)
    tasks = distribute_tasks(tasks, rank, size)

    if not os.path.isdir(processed_data_folder):
        os.mkdir(processed_data_folder)
    for dirpath, dirs, files in tasks:
        for filename in files:
            fname = os.path.join(dirpath,filename)
            if fname.endswith('.npz'):
                print(fname)
                #process_file(fname, processed_data_folder, mods=["obs", "acts"], smoothing=5)
                #process_file(fname, processed_data_folder, mods=["obs_cont", "disc_cond"], smoothing=5)
                #process_file(fname, processed_data_folder, mods=["disc_cond"], smoothing=5)
                process_file(fname, processed_data_folder, mods=["obs" ,"acts", "obs_cont", "disc_cond"], smoothing=5)
