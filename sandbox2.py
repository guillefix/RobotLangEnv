import numpy as np

import glob

# files = glob.glob("/home/guillefix/code/inria/UR5/**/**/**/*.npz")
files = glob.glob("/home/guillefix/code/inria/UR5_new/**/**/**/*.npz")

from src.envs.env_params import get_env_params

env_params = get_env_params()
object_types = env_params['types']

import pickle

pickle.dump(object_types, open("object_types.pkl", "wb"))


for f in files:
    a = dict(np.load(f, allow_pickle=True))
    for i in range(3):
        type_encoding = np.zeros([len(object_types)])
        object_type = a['obj_stuff'][0][i]['type']
        type_encoding[object_types.index(object_type)] = 1
        a['obs'][:,14 + i * 35: 37 + i * 35] = type_encoding
    np.savez(f, **a)


# a = dict(np.load(files[0], allow_pickle=True))
# type_encoding
# a['obs'][:,14 + i * 35: 37 + i * 35] = type_encoding
# a['obs'][0,14 + i * 35: 37 + i * 35]

type = object_types[2]
object_types.index(type)
for f in files:
    a = np.load(f, allow_pickle=True)
    obs = a['obs'][0]
    # print(obs)
    object_indices = []
    for i in range(3):
        otype_enc =  obs[14 + i * 35: 37 + i * 35]
        # col =  obs[37 + i * 35: 40 + i * 35]
        # col = infer_color(col)
        # print(otype_enc)
        # otype_enc = np.zeros([len(object_types)])
        # object_type = a['obj_stuff'][0][i]['type']
        # otype_enc[object_types.index(object_type)] = 1
        otype_index = np.argwhere(otype_enc == 1.0)[0][0]
        object_indices.append(otype_index)
        # print(otype_index)
    for i,obj in enumerate(a['obj_stuff'][0]):
        if obj['type'] == type:
            print(type)
            # print(otype_enc)
            print(object_indices[i])
