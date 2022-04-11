import numpy as np
import os
from create_simple_dataset import has_concrete_object_ann, check_if_exact_one_object_obs, get_new_obs_obs
from src.envs.color_generation import infer_color
from constants import *

import pickle
import json
object_types = pickle.load(open(root_folder+"object_types.pkl","rb"))
vocab_default=json.load(open(processed_data_folder+"acts.npy.annotation.class_index.json", "r"))

def get_tokens(goal_str, input_lengths=[11], obj_stuff=None, vocab=None):
    if vocab==None:
        vocab = vocab_default
    tokens = []
    words = goal_str.split(" ")
    for i in range(11):
        if i < len(words):
            word = words[i]
            tokens.append(vocab[word])
        else:
            tokens.append(66)

    tokens = np.array(tokens)
    # discrete_input = np.load(data_folder+filename+"."+input_mods[0]+".npy")
    if input_lengths[0] == 10:
        tokens = np.concatenate([tokens[:1], tokens[2:]])
    elif input_lengths[0] == 14:
        obj_types = [vocab[t] for t in map(lambda x: x["type"], obj_stuff)]
        obj_types = np.array(obj_types)
        tokens = np.concatenate([tokens, obj_types])

    return tokens

def get_obj_types(obss, vocab=None):
    if vocab==None:
        vocab = vocab_default
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

def get_ann_with_obj_types(ann, obss):
    obj_types = get_obj_types(obss)
    disc_cond = np.concatenate([ann, obj_types])
    return disc_cond

def get_obs_cont_single(obs):
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


def get_obs_cont(obss):
    if len(obbs.shape) == 2:
        return get_obs_cont_single(obss)
    obss_color1 = obss[:,37:40]
    obss_color2 = obss[:,72:75]
    obss_color3 = obss[:,107:110]
    n=len(color_list)
    obss_color1 = np.stack([one_hot(color_list.index(infer_color(x)),n) for x in obss_color1])
    obss_color2 = np.stack([one_hot(color_list.index(infer_color(x)),n) for x in obss_color2])
    obss_color3 = np.stack([one_hot(color_list.index(infer_color(x)),n) for x in obss_color3])
    obss_disc1 = np.argmax(obss[:,14:37], axis=1)
    obss_disc2 = np.argmax(obss[:,49:72], axis=1)
    obss_disc3 = np.argmax(obss[:,84:107], axis=1)
    obss_cont = np.concatenate([obss[:,:14], obss_color1, obss[:,40:49], obss_color2, obss[:,75:84], obss_color3, obss[:,110:]],axis=1)
    return obs_cont

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
