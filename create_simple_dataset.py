import numpy as np
import os
from constants import *
import pickle

import json
object_types = pickle.load(open(root_folder+"object_types.pkl","rb"))
vocab=json.load(open(processed_data_folder+"npz.annotation.txt.annotation.class_index_reverse.json", "r"))
print(vocab)

def has_concrete_object(filename):
    annotation_file = processed_data_folder+filename+".npz.annotation.txt"
    ann = open(annotation_file, "r").read()
    return has_concrete_object_ann(ann)

def has_concrete_object_ann(ann):
    ann_arr = ann.split(" ")
    action = ann_arr[0]
    adjective = ann_arr[1]
    object = ann_arr[2]
    if adjective in color_list and object in object_types:
        return True, adjective, object
    else:
        return False, None, None


def check_if_exact_one_object(filename, color, object_type):
    disc_cond_file = processed_data_folder+filename+".npz.disc_cond.npy"
    obs_file = processed_data_folder+filename+".npz.obs_cont.npy"
    obs = np.load(obs_file)
    disc_cond = np.load(disc_cond_file)

    return check_if_exact_one_object_from_obs(obs, disc_cond, color, object_type)

def check_if_exact_one_object_from_obs(obs, disc_cond, color, object_type):
    col1 = color_list[np.argmax(obs[0,14:22])]
    col2 = color_list[np.argmax(obs[0,31:39])]
    col3 = color_list[np.argmax(obs[0,48:56])]
    obj_cols = [col1, col2, col3]
    # print(obj_cols)

    type1 = vocab[str(int(disc_cond[-3]))]
    type2 = vocab[str(int(disc_cond[-2]))]
    type3 = vocab[str(int(disc_cond[-1]))]
    obj_types = [type1, type2, type3]
    # print(obj_types)

    matches = 0
    index_first_match = -1
    for i in range(3):
        if color == obj_cols[i] and object_type == obj_types[i]:
            matches += 1
            index_first_match = i

    return matches == 1, index_first_match

def get_new_obs(filename, obj_index, nocol=False, noarm=False):
    obs_file = processed_data_folder+filename+".npz.obs_cont.npy"
    obs = np.load(obs_file)
    return get_new_obs_from_obs(obs, obj_index, nocol=nocol, noarm=noarm)

def get_new_obs_from_obs(obs, obj_index, nocol=False, noarm=False):
    arm_obs = obs[:,:8]
    obj_data = obs[:,8+(9+8)*obj_index:8+(9+8)*(obj_index+1)]
    obj_pos_orn = obj_data[:,:6]
    obj_col = obj_data[:,6:6+8]
    extra_obs = obs[:,8+(9+8)*3:]

    if nocol:
        if noarm:
            new_obs = np.concatenate([obj_pos_orn, extra_obs], axis=1)
        else:
            new_obs = np.concatenate([arm_obs, obj_pos_orn, extra_obs], axis=1)
    else:
        if noarm:
            new_obs = np.concatenate([obj_pos_orn, obj_col, extra_obs], axis=1)
        else:
            new_obs = np.concatenate([arm_obs, obj_pos_orn, obj_col, extra_obs], axis=1)
    return new_obs

#%%

# get_new_obs(filenames[0], 0).shape

if __name__ == "__main__":
    filenames=[x[:-1] for x in open(processed_data_folder+"base_filenames.txt","r").readlines()]

    new_base_filenames_file = open(processed_data_folder+"base_filenames_single_objs.txt", "w")
    # paint_anns = []
    # all_anns = []

    for filename in filenames:
        # if not ("Guillermo" in filename or "Tianwei" in filename):
        #     continue

        has_conc_obj, color, object_type = has_concrete_object(filename)
        annotation_file = processed_data_folder+filename+".npz.annotation.txt"
        ann = open(annotation_file, "r").read()
        # all_anns.append(ann)
        # if "Paint" in ann: paint_anns.append(ann)

        if has_conc_obj:
            exact_one_object, obj_index = check_if_exact_one_object(filename, color, object_type)
            if exact_one_object:
                print(filename)
                new_obs = get_new_obs(filename, obj_index, nocol=True, noarm=True)
                new_base_filenames_file.write(filename+"\n")

                # np.save(processed_data_folder+filename+".obs_cont_single.npy", new_obs)
                ann_arr = np.load(processed_data_folder+filename+".npz.annotation.txt.annotation.npy")
                # ann_arr_simp = np.concatenate([ann_arr[:1], ann_arr[3:]])
                ann_arr_simp = np.concatenate([ann_arr[:1], ann_arr[2:]])
                np.save(processed_data_folder+filename+".annotation_simp.npy", ann_arr_simp)
                # np.save(processed_data_folder+filename+".annotation_simp_wnoun.npy", ann_arr_simp)
                # np.save(processed_data_folder+filename+".obs_cont_single.npy", new_obs)
                # np.save(processed_data_folder+filename+".obs_cont_single_nocol.npy", new_obs)
                np.save(processed_data_folder+filename+".obs_cont_single_nocol_noarm.npy", new_obs)

    new_base_filenames_file.close()


# "Guillermo" in "UR5_Guillermo_obs"

    # len(all_anns)
    # len(paint_anns)
