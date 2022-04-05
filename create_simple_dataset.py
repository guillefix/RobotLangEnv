import numpy as np
root_folder="/home/guillefix/code/inria/UR5_processed/"
filenames=[x[:-1] for x in open(root_folder+"base_filenames.txt","r").readlines()]


# actions = ('Open', 'Close', 'Grasp', 'Put', 'Hide', 'Turn on', 'Turn off', 'Make', 'Paint', 'Move', 'Throw')
actions_with_object = ('Grasp', 'Put', 'Hide', 'Paint', 'Move')
# colors = list(('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'black'))
colors = ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']
positions = ('on the left side of the table', 'on the right side of the table', 'on the shelf', 'behind the door', 'in the drawer')

# list objects and categories
geometric_solid = ('cube', 'block', 'cylinder')
kitchen_ware = ('bottle', 'bowl', 'plate', 'cup', 'spoon')
animal_model = ('bear', 'bird', 'dog', 'fish', 'elephant')
food_model = ('apple', 'banana', 'cookie', 'donut', 'sandwich')
vehicles_model = ('train', 'plane', 'car', 'bike', 'bus')

categories = dict(solid = geometric_solid,
                  kitchenware = kitchen_ware,
                  animal = animal_model,
                  food = food_model,
                  vehicle = vehicles_model,
                  )
# List types
types = ()
for k_c in categories.keys():
    types += categories[k_c]
types = tuple(set(types)) # filters doubles, when some categories include others.
nb_types = len(types)

def has_concrete_object(filename):
    annotation_file = root_folder+filename+".annotation.txt"
    ann = open(annotation_file, "r").read()
    return has_concrete_object_ann(ann)

def has_concrete_object_ann(ann):
    ann_arr = ann.split(" ")
    action = ann_arr[0]
    adjective = ann_arr[1]
    object = ann_arr[2]
    if adjective in colors and object in types:
        return True, adjective, object
    else:
        return False, None, None

import json
vocab = json.loads(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index_reverse.json","r").read())

def check_if_exact_one_object(filename, color, object_type):
    disc_cond_file = root_folder+filename+".disc_cond.npy"
    obs_file = root_folder+filename+".obs_cont.npy"
    obs = np.load(obs_file)
    disc_cond = np.load(disc_cond_file)

    return check_if_exact_one_object_obs(obs, disc_cond, color, object_type)

def check_if_exact_one_object_obs(obs, disc_cond, color, object_type):
    col1 = colors[np.argmax(obs[0,14:22])]
    col2 = colors[np.argmax(obs[0,31:39])]
    col3 = colors[np.argmax(obs[0,48:56])]
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
    obs_file = root_folder+filename+".obs_cont.npy"
    obs = np.load(obs_file)
    return get_new_obs_obs(obs, obj_index, nocol=nocol, noarm=noarm)

def get_new_obs_obs(obs, obj_index, nocol=False, noarm=False):
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

    new_base_filenames_file = open(root_folder+"base_filenames_single_objs.txt", "w")
    paint_anns = []
    all_anns = []

    for filename in filenames:
        # if not ("Guillermo" in filename or "Tianwei" in filename):
        #     continue

        has_conc_obj, color, object_type = has_concrete_object(filename)
        annotation_file = root_folder+filename+".annotation.txt"
        ann = open(annotation_file, "r").read()
        all_anns.append(ann)
        if "Paint" in ann: paint_anns.append(ann)

        if has_conc_obj:
            exact_one_object, obj_index = check_if_exact_one_object(filename, color, object_type)
            if exact_one_object:
                print(filename)
                new_obs = get_new_obs(filename, obj_index, nocol=True, noarm=True)
                # new_base_filenames_file.write(filename+"\n")

                np.save(root_folder+filename+".obs_cont_single.npy", new_obs)
                ann_arr = np.load(root_folder+filename+".annotation.npy")
                # ann_arr_simp = np.concatenate([ann_arr[:1], ann_arr[3:]])
                ann_arr_simp = np.concatenate([ann_arr[:1], ann_arr[2:]])
                # np.save(root_folder+filename+".annotation_simp.npy", ann_arr_simp)
                # np.save(root_folder+filename+".annotation_simp_wnoun.npy", ann_arr_simp)
                # np.save(root_folder+filename+".obs_cont_single.npy", new_obs)
                # np.save(root_folder+filename+".obs_cont_single_nocol.npy", new_obs)
                # np.save(root_folder+filename+".obs_cont_single_nocol_noarm.npy", new_obs)

    new_base_filenames_file.close()


# "Guillermo" in "UR5_Guillermo_obs"

    len(all_anns)
    len(paint_anns)
