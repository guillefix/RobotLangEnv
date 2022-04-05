import numpy as np
import time
import torch.nn.functional as F

from src.envs.env_params import get_env_params

color_list = ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']

from src.envs.color_generation import infer_color

def one_hot(x,n):
    a = np.zeros(n)
    a[x]=1
    return a

env_params = get_env_params()
env_params['types']

# root_folder = "/home/guillefix/code/multimodal-transflower/"
root_folder="/home/guillefix/code/inria/UR5_processed/"

# import importlib.util
# spec = importlib.util.spec_from_file_location("module", root_folder+"inference/generate.py")
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)
# load_model_from_logs_path = module.load_model_from_logs_path
#
# # experiment = "transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw3"
# experiment = "moglow_expmap1_tw"
# model, opt = load_model_from_logs_path(root_folder+"training/experiments/"+experiment)


# session_id="Alex"
# rec_id="4"
# rec_id="8"
# session_id="Laetitia"
# rec_id="17"
# session_id="Gautier" # seems okay
# rec_id="5"
# rec_id="9"
# rec_id="10"
# session_id="Masataka"
# rec_id="1"
# rec_id="3"
# rec_id="4"
# rec_id="5"
# session_id="Mathieu"
# rec_id="11"
# rec_id="12"
# session_id="Rania"
# rec_id="1"
# rec_id="3"
# rec_id="5"
# session_id="Smeety"
# rec_id="2"
# rec_id="3"
# rec_id="4"
# rec_id="5"
# session_id="Thomas"
# rec_id="3"
# rec_id="5"
# rec_id="9"
# session_id="Tianwei"
# rec_id="13"
# rec_id="14"
# rec_id="15"
# rec_id="19"
# session_id="Tianwei2"
# rec_id="4"
# rec_id="5"
# session_id="Tianwei3"
# rec_id="7"
# rec_id="8"
# session_id="Tianwei5"
# rec_id="1"
session_id="Guillermo"
# rec_id="8"
rec_id="55"
# a = np.load("/home/guillefix/code/inria/UR5/Guillermo/obs_act_etc/8/data.npz", allow_pickle=True)
# a = np.load("/home/guillefix/code/inria/UR5/Guillermo/obs_act_etc/24/data.npz", allow_pickle=True)
# a = np.load("/home/guillefix/code/inria/UR5/Guillermo/obs_act_etc/13/data.npz", allow_pickle=True)
# a = np.load("/home/guillefix/code/inria/UR5/Guillermo/obs_act_etc/6/data.npz", allow_pickle=True)
# a = np.load("/home/guillefix/code/inria/UR5/Tianwei2/obs_act_etc/98/data.npz", allow_pickle=True)
# a = np.load("/home/guillefix/code/inria/UR5/Tianwei/obs_act_etc/24/data.npz", allow_pickle=True)
# a = np.load("/home/guillefix/code/inria/UR5/Tianwei/obs_act_etc/3/data.npz", allow_pickle=True)
# a = np.load("/home/guillefix/code/inria/UR5/Tianwei/obs_act_etc/4/data.npz", allow_pickle=True)
a = np.load("/home/guillefix/code/inria/UR5/"+session_id+"/obs_act_etc/"+rec_id+"/data.npz", allow_pickle=True)
n=2
na=2
# n=10
# na=10
# na=5
# n=2
# n=3
# n=10
# n=30
# n=120
# n=60
nocol=True
noarm=True
# nocol=False
# noarm=False

from create_simple_dataset import has_concrete_object_ann, check_if_exact_one_object_obs, get_new_obs_obs

a = dict(a)

list(a.keys())
a['obj_stuff']
import matplotlib.pyplot as plt

#%%
a["obs"].shape


print(a['obs'].shape)
goal_str = str(a['goal_str'][0])
# print(goal_str)
a['goal_str'] = np.array([goal_str], dtype='O')
#%%

# goal_str = 'Grasp black bear'
# goal_str = 'Put white dog on the shelf'
# goal_str = 'Put cyan apple on the shelf'
# goal_str = 'Paint yellow fish blue'
# goal_str = 'Put white dog on the shelf'
# goal_str = 'Put green plate on the shelf'
# goal_str = 'Paint black bird yellow'
# goal_str = 'Put blue bird on the shelf'
# goal_str = 'Hide blue dog'
# goal_str = 'Put blue dog behind the door'
# goal_str = 'Put blue dog in the drawer'
# goal_str = 'Put blue dog in the drawer'
# goal_str = 'Put black donut in the drawer'
# goal_str = 'Paint black donut green'
# goal_str = 'Paint red spoon green'
# goal_str = 'Put yellow car in the drawer'
# goal_str = 'Put magenta bike on the shelf'
# goal_str = 'Put white dog on the shelf'
# goal_str = 'Paint green plate red'
print(goal_str)
a['goal_str'][0] = goal_str
# print('-')
# # b = np.array(["ab"], dtype='O')
# b = np.array(["ab"])
# print(b.dtype)
# b[0] = "apple"
# print(b[0])
# print('-')
# print(a['goal_str'][0])
# a['gripper_proprioception']
# a['obj_stuff']

# a['acts'][0][0]
# a['acts_rpy_rel']
# plt.matshow(a['acts'][:100])
# plt.matshow(a['acts'][:100])
# a['acts']


# acts = np.load("/home/guillefix/code/multimodal-transflower/inference/generated/transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw/predicted_mods/UR5_Tianwei2_obs_act_etc_98_data.npz.acts_scaled.generated.npy")
# acts = np.load("/home/guillefix/code/multimodal-transflower/inference/generated/transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw/predicted_mods/UR5_Tianwei_obs_act_etc_2_data.npz.acts_scaled.generated.npy")
# acts = np.load("/home/guillefix/code/multimodal-transflower/inference/generated/transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw/predicted_mods/UR5_Tianwei_obs_act_etc_0_data.npz.acts_scaled.generated.npy")
# acts = np.load("/home/guillefix/code/multimodal-transflower/inference/generated/transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw3/predicted_mods/UR5_Tianwei_obs_act_etc_1_data.npz.acts_scaled.generated.npy")
# acts = np.load("/home/guillefix/code/multimodal-transflower/inference/generated/transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw3/predicted_mods/UR5_Tianwei_obs_act_etc_4_data.npz.acts_scaled.generated.npy")
# acts = acts[:,0,:]
# a['acts'] = acts
# a["acts"].shape
# acts.shape

#%%

import pickle
root_folder2="/home/guillefix/code/inria/"
# acts_scaler = pickle.load(open(root_folder2+"UR5_processed/acts_scaled_scaler.pkl", "rb"))
acts_scaler = pickle.load(open(root_folder2+"UR5_processed/acts_trim_scaled_scaler.pkl", "rb"))
# obs_scaler = pickle.load(open(root_folder2+"UR5_processed/obs_scaled_scaler.pkl", "rb"))
# obs_scaler = pickle.load(open(root_folder2+"UR5_processed/obs_cont_scaled_scaler.pkl", "rb"))
if nocol and not noarm:
    obs_scaler = pickle.load(open(root_folder2+"UR5_processed/obs_cont_single_nocol_scaled_scaler.pkl", "rb"))
elif nocol and noarm:
    # obs_scaler = pickle.load(open(root_folder2+"UR5_processed/obs_cont_single_nocol_noarm_scaled_scaler.pkl", "rb"))
    obs_scaler = pickle.load(open(root_folder2+"UR5_processed/obs_cont_single_nocol_noarm_trim_scaled_scaler.pkl", "rb"))
else:
    obs_scaler = pickle.load(open(root_folder2+"UR5_processed/obs_cont_single_scaled_scaler.pkl", "rb"))


#%%

#LOAD MODEL

import torch
# model = torch.jit.load('compiled_jit.pth', map_location=torch.device('cpu'))
# model = torch.jit.load('compiled_jit.pth', map_location=torch.device('cuda:0'))
# model = torch.jit.load('compiled_jit.pth')
# model = torch.jit.load('compiled_jit2.pth')
# model = torch.jit.load('compiled_jit3.pth')
# model = torch.jit.load('compiled_jit4.pth')
# model = torch.jit.load('compiled_jit5.pth')

import sys
root_dir = "/home/guillefix/code/multimodal-transflower"
sys.path.append(root_dir)

# from inference.generate import load_model_from_logs_path


default_save_path = "pretrained/transflower_zp5_single_obj_nocol_trim_tw_single_filtered"
# default_save_path = "pretrained/transflower_zp5_long_single_obj_nocol_trim_tw_single_filtered"
logs_path = default_save_path
#load model:
# model, opt = load_model_from_logs_path(logs_path)


torch.__version__

import json

d=json.load(open("UR5_processed/acts.npy.annotation.class_index.json", "r"))

tokens = []
words = goal_str.split(" ")
for i in range(11):
    if i < len(words):
        word = words[i]
        tokens.append(d[word])
    else:
        tokens.append(66)

import pickle
object_types = pickle.load(open("/home/guillefix/code/inria/captionRLenv/object_types.pkl","rb"))
# object_types

# import json
# vocab = json.loads(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index.json","r").read())
# obss_disc1 = np.argmax(a['obs'][:,14:37], axis=1)
# obss_disc2 = np.argmax(a['obs'][:,49:72], axis=1)
# obss_disc3 = np.argmax(a['obs'][:,84:107], axis=1)
# assert np.all(obss_disc1 == obss_disc1[0])
# assert np.all(obss_disc2 == obss_disc2[0])
# assert np.all(obss_disc3 == obss_disc3[0])
# obss_disc1 = obss_disc1[0]
# obss_disc2 = obss_disc2[0]
# obss_disc3 = obss_disc3[0]
# obss_disc1 = vocab[object_types[obss_disc1]]
# obss_disc2 = vocab[object_types[obss_disc2]]
# obss_disc3 = vocab[object_types[obss_disc3]]
#
# tokens_expanded = tokens + [obss_disc1, obss_disc2, obss_disc3]
#
# disc_cond = np.array(tokens_expanded)
# tokens = np.array(tokens)
# print(tokens.shape)

# filename = "UR5_Guillermo_obs_act_etc_8_data"
# filename = "UR5_Guillermo_obs_act_etc_24_data"
# filename = "UR5_Tianwei_obs_act_etc_24_data"
filename = "UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data"
# obs_cont = np.load(root_folder+filename+".obs_cont.npy")
obs_cont = np.load(root_folder+filename+".obs_cont.npy")
# act_data = np.load(root_folder+filename+".npz.acts.npy")
act_data = np.load(root_folder+filename+".acts_trim.npy")
# act_data[:,3:7] = fix_quaternions(act_data[:,3:7])

# import matplotlib.pyplot as plt
#
# act_data[:10,3]
# import scipy.ndimage.filters as filters
# act_data_smooth1 = filters.gaussian_filter1d(act_data, 2, mode='nearest', axis=0)
# # act_data_smooth = filters.maximum_filter1d(act_data, 5, mode='nearest', axis=0)
# act_data_smooth2 = filters.uniform_filter1d(act_data, 5, mode='nearest', axis=0)
# # %matplotlib
# plt.plot(act_data[:100,4])
# plt.plot(act_data_smooth1[:100,4])
# plt.plot(act_data_smooth2[:100,4])
# a['acts'][:,3:7] = fix_quaternions(a['acts'][:,3:7])
# plt.plot(a['acts'][:100,3])

disc_cond = np.load(root_folder+filename+".disc_cond.npy")
# tokens = np.load(root_folder+filename+".annotation.npy")
if nocol:
    tokens = np.concatenate([tokens[:1], tokens[2:]])
# tokens = np.array(tokens)
# tokens = disc_cond

has_conc_obj, color, object_type = has_concrete_object_ann(goal_str)
print(color, object_type)
# assert has_conc_obj
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
exact_one_object, obj_index = check_if_exact_one_object_obs(obs_cont, disc_cond, color, object_type)
# assert exact_one_object


# prev_obs = a['obs'][:120]
# prev_acts = a['acts'][:120]
# prev_obs = obs_scaler.inverse_transform(np.zeros((n,125)))
# prev_acts = acts_scaler.inverse_transform(np.zeros((n,8)))
# prev_obs = obs_scaler.inverse_transform(np.zeros((n,71)))
# prev_obs = obs_scaler.inverse_transform(np.zeros((n,34)))
# prev_obs = obs_scaler.inverse_transform(np.zeros((n,26)))
prev_obs = obs_scaler.inverse_transform(np.zeros((n,18)))
prev_acts = acts_scaler.inverse_transform(np.zeros((na,8)))

# prev_acts2 = np.concatenate([prev_acts,a['acts']])
# prev_obs2 = np.concatenate([prev_obs,a['obs']])

# prev_obs = a['obs'][:n]
# prev_obs = np.stack([get_obs_cont(o) for o in prev_obs])
# prev_obs = get_new_obs_obs(prev_obs, obj_index, nocol=nocol, noarm=noarm)
# prev_acts = a['acts'][:na]
prev_acts2 = a['acts'][:20]

# prev_obs2 = a['obs']
# prev_obs = prev_obs2[:n]
# prev_acts = prev_acts2[n:2*n]
# prev_obs = np.concatenate([prev_obs,a['obs']])[:120]
# prev_acts = np.concatenate([prev_acts,a['acts']])[:120]
# prev_obs = a['obs'][:120]
# prev_acts = a['acts'][40:160]
# acts_scaler.transform(prev_acts).shape

from data_utils import fix_quaternions

def make_inputs(tokens, prev_obs, prev_acts):
    # return [torch.from_numpy(tokens.copy()).unsqueeze(1).unsqueeze(1).cuda(), torch.from_numpy(prev_obs.copy()).unsqueeze(1).float().cuda(), torch.from_numpy(prev_acts.copy()).unsqueeze(1).float().cuda()]
    # prev_obs[3:7] = fix_quaternions(prev_obs[3:7])
    prev_obs = obs_scaler.transform(prev_obs)
    prev_acts[3:7] = fix_quaternions(prev_acts[3:7])
    prev_acts = acts_scaler.transform(prev_acts)
    tokens = torch.from_numpy(tokens)
    # tokens = F.one_hot(tokens,num_classes=67)
    tokens = tokens.unsqueeze(1).unsqueeze(1).long().cuda()
    # tokens = tokens.unsqueeze(1).cuda()
    return [tokens, torch.from_numpy(prev_obs).unsqueeze(1).float().cuda(), torch.from_numpy(prev_acts).unsqueeze(1).float().cuda()]
    # return [torch.from_numpy(tokens).unsqueeze(1).unsqueeze(1).cpu(), torch.from_numpy(prev_obs).unsqueeze(1).float().cpu(), torch.from_numpy(prev_acts).unsqueeze(1).float().cpu()]

inputs = make_inputs(tokens, prev_obs, prev_acts)
# inputs = [torch.from_numpy(tokens.copy()).unsqueeze(1).unsqueeze(1), torch.from_numpy(init_obs.copy()).unsqueeze(1).float(), torch.from_numpy(init_acts.copy()).unsqueeze(1).float()]

# z_shape = (1,8,1)
# eps_std = 1.0
# noises = [torch.normal(mean=torch.zeros(z_shape), std=torch.ones(z_shape)*eps_std).cuda()]

# import os
# os.environ["PYTORCH_JIT_LOG_LEVEL"]='profiling_graph_executor_impl'

# out = model(inputs, noises)
# out = model(inputs, noises)
# latent = torch.randn((251,1,800)).cuda()
# while True:
#     # try:
#     print("HI")
#     # out = model(inputs, noises)
#     out = model(latent)
#     print(out)
#     # except:
#     #     time.sleep(1)
#     #     continue

print(inputs[0].shape)
print(inputs[1].shape)
print(inputs[2].shape)

##NEED THIS TO PREPARE MODEL when using TorchScript
# out = model(inputs)
# print(out)
# out = model(inputs)
# print(out)

# out.shape
# out[0].shape

#%%

import os
os.chdir("./captionRLenv/")
from src.envs.envList import *
import numpy as np
import pybullet as p
import pickle as pk

def add_xyz_rpy_controls(env):
    controls = []
    orn = env.instance.default_arm_orn_RPY
    controls.append(env.p.addUserDebugParameter("X", -1, 1, 0))
    controls.append(env.p.addUserDebugParameter("Y", -1, 1, 0.00))
    controls.append(env.p.addUserDebugParameter("Z", -1, 1, 0.2))
    controls.append(env.p.addUserDebugParameter("R", -4, 4, orn[0]))
    controls.append(env.p.addUserDebugParameter("P", -4, 4, orn[1]))
    controls.append(env.p.addUserDebugParameter("Y", -4,4, orn[2]))
    controls.append(env.p.addUserDebugParameter("grip", env.action_space.low[-1], env.action_space.high[-1], 0))
    return controls

def add_joint_controls(env):
    for i, obj in enumerate(env.instance.restJointPositions):
        env.p.addUserDebugParameter(str(i), -2*np.pi, 2*np.pi, obj)


joint_control = False # Toggle this flag to control joints or ABS RPY Space
env = UR5PlayAbsRPY1Obj()

import pickle
object_types = pickle.load(open("object_types.pkl","rb"))
env.env_params['types'] = object_types

#%%
from src.envs.descriptions import generate_all_descriptions
from src.envs.env_params import get_env_params
# env_params = get_env_params()
# all_descriptions = generate_all_descriptions(env_params)[2]
# while True:
#     d = np.random.choice(all_descriptions)
#     if d.lower().split()[0] not in ['turn', 'open', 'close', 'make']:
#         break
# print(d)
env.render(mode='human')
# print(a["goal_str"][0])
import pickle
# types = pickle.load(open("object_types.pkl","rb"))
# i=2
# a['obj_stuff'][0] = {'type': types[i], 'color': 'white', 'category': None}
# goal_str_arr = goal_str.split(" ")
# goal_str_arr[1] = 'white'
# goal_str_arr[2] = types[i]
# goal_str = " ".join(goal_str_arr)
env.reset(o=a["obs"][0], info_reset=None, description=goal_str, joint_poses=a["joint_poses"][0], objects=a['obj_stuff'][0], restore_objs=True)
# env.reset(o=a["obs"][0], info_reset=None, description=a["goal_str"][0], joint_poses=None, objects=None)

from src.envs.reward_function import get_reward_from_state



print([o for o in env.instance.objects])
if joint_control:
    add_joint_controls(env)
else:
    controls = add_xyz_rpy_controls(env)


temp=1.0
# temp_delta=0.95
temp_delta=0.99
for i in range(1000000):

    if joint_control:
        poses  = []
        for i in range(len(env.instance.restJointPositions)):
            poses.append(env.p.readUserDebugParameter(i))
        # Uses a hard reset of the arm joints so that we can quickly debug without worrying about forces
        env.instance.reset_arm_joints(env.instance.arm, poses)

    else:
        # acts = a['acts_rpy'][i]
        # action = acts

        acts = a['acts'][i]
        # acts = a['acts'][0]

        # acts = model(inputs, temp=0.3)[0][0].cpu()
        # acts = model(inputs)[0][0].cpu()

        # acts = model(inputs)[0][0][0].cpu()
        # variance = np.mean(np.abs(prev_acts2[0]-prev_acts2[-1]))
        variance = np.max(np.abs(prev_acts2[0]-prev_acts2[-1]))
        # print(variance)
        # temp = temp*temp_delta +(1-temp_delta)*20*np.tanh(0.01/variance)
        # temp = temp*temp_delta +(1-temp_delta)*10*np.tanh(0.01/variance)
        temp = np.max([temp*temp_delta +(1-temp_delta)*10*np.tanh(0.01/variance), 0.5])
        # temp = temp*temp_delta +(1-temp_delta)*2*np.tanh(0.01/variance)
        # temp = 0.5
        # temp = 1.0
        # temp = 0.1
        # print(temp)
        # acts = model(inputs, temp=temp)[0][0][0].cpu()
        # acts = acts_scaler.inverse_transform(acts)
        # acts = acts[0]
        # print(acts.shape)

        action = [acts[0],acts[1],acts[2]] + list(p.getEulerFromQuaternion(acts[3:7])) + [acts[7]]
        # action = [acts[0],acts[1],acts[2]] + [0,0,0] + [acts[7]]
        # action = []
        # for j,control in enumerate(controls):
            # action.append(env.p.readUserDebugParameter(control))
            # action.append(np.random.rand())
            # action.append(acts)

        # print(action)
        state = env.instance.calc_actor_state()
        obs, r, done, info = env.step(np.array(action))
        # obs[8] = 0.15
        # obs[9] = 0.0
        # choices = [np.array([0.15,0]),np.array([-0.15,0]),np.array([0.15,0.30]),np.array([-0.15,0.30])]
        # obs[8:10] = choices[i%len(choices)]
        # # obs[10:13] = [0,0,0]
        # size = obs[40:43]
        # max_size = np.max(size)
        # # if i<100:
        # #     obs[8:11] = [0,0.15,max_size/2]
        # env.instance.reset_objects(obs)
        # print(obs[8:11])
        # print(infer_color(obs[37:40]))
        # print(env.instance.state_dict)
        if i == 0:
            initial_state = obs
        else:
            current_state = obs
            print(get_reward_from_state(initial_state, current_state, a['obj_stuff'], goal_str, env.instance.env_params))

        obs_cont = get_obs_cont(obs)
        new_obs = get_new_obs_obs(obs_cont[None], obj_index, nocol=nocol, noarm=noarm)
        # print(new_obs)

        prev_acts = np.concatenate([prev_acts[1:],acts[None]])
        prev_acts2 = np.concatenate([prev_acts2[1:],acts[None]])
        # prev_acts = prev_acts2[i:n+i]
        # print(acts[None].shape)
        # prev_obs = np.concatenate([prev_obs[:-1],obs[None]])
        # prev_obs = np.concatenate([prev_obs[:-1],obs_cont[None]])

        # prev_obs = np.concatenate([prev_obs[1:],new_obs])

        # for j in range(3):
        #     prev_obs[:,14 + j * 35: 37 + j * 35] = prev_obs2[:n,14 + j * 35: 37 + j * 35]
        # prev_obs = prev_obs2[i:n+i]
        # print(obs[None].shape)
        inputs = make_inputs(tokens, prev_obs, prev_acts)
        # print(inputs[0].shape)
        # print(inputs[1].shape)
        # print(inputs[2].shape)
