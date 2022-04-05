import numpy as np
from create_simple_dataset import has_concrete_object_ann, check_if_exact_one_object_obs, get_new_obs_obs
color_list = ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']
from src.envs.color_generation import infer_color

def get_obs_cont(obs):
    obs_color1 = obs[37:40].astype(np.float64)
    obs_color2 = obs[72:75].astype(np.float64)
    obs_color3 = obs[107:110].astype(np.float64)
    # print(obs_color1, obs_color2, obs_color3)
    n=len(color_list)
    obs_color1 = one_hot(color_list.index(infer_color(obs_color1)),n)
    obs_color2 = one_hot(color_list.index(infer_color(obs_color2)),n)
    obs_color3 = one_hot(color_list.index(infer_color(obs_color3)),n)
    obs_cont = np.concatenate([obs[:14], obs_color1, obs[40:49], obs_color2, obs[75:84], obs_color3, obs[110:]])
    return obs_cont


def get_obj_types(obss):
    obss_disc1 = np.argmax(obss[:,14:37], axis=1)
    obss_disc2 = np.argmax(obss[:,49:72], axis=1)
    obss_disc3 = np.argmax(obss[:,84:107], axis=1)

    assert np.all(obss_disc1 == obss_disc1[0])
    assert np.all(obss_disc2 == obss_disc2[0])
    assert np.all(obss_disc3 == obss_disc3[0])
    # assert np.all(obss_color1 == obss_color1[0])
    # assert np.all(obss_color2 == obss_color2[0])
    # assert np.all(obss_color3 == obss_color3[0])
    obss_disc1 = obss_disc1[0]
    obss_disc2 = obss_disc2[0]
    obss_disc3 = obss_disc3[0]
    # obss_color1 = obss_color1[0]
    # obss_color2 = obss_color2[0]
    # obss_color3 = obss_color3[0]

    obss_disc1 = vocab[object_types[obss_disc1]]
    obss_disc2 = vocab[object_types[obss_disc2]]
    obss_disc3 = vocab[object_types[obss_disc3]]

    # disc_cond = np.concatenate([ann, [obss_disc1, obss_color1, obss_disc2, obss_color2, obss_disc3, obss_color3]])
    # disc_cond = np.concatenate([ann, [obss_disc1, obss_disc2, obss_disc3]])
    obj_types = np.array([obss_disc1, obss_disc2, obss_disc3])
    return obj_types

def fix_quaternions(rot_stream):
    prev_rot = None
    for i, rot in enumerate(rot_stream):
        if prev_rot is None:
            prev_rot = rot
        if np.any((np.abs(rot-prev_rot) >= np.abs(prev_rot)) * (np.abs(rot-prev_rot)>=5e-2)):
            rot_stream[i:] = -rot_stream[i:]
        prev_rot = rot_stream[i]

    return rot_stream

def one_hot(x,n):
    a = np.zeros(n)
    a[x]=1
    return a
