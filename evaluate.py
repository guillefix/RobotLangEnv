import numpy as np
import time
import torch.nn.functional as F
import pickle
from pathlib import Path
import os

from src.envs.env_params import get_env_params
from src.envs.color_generation import infer_color
from utils.data_utils import get_obs_cont, get_obj_types, fix_quaternions, one_hot

color_list = ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']

if "ROOT_FOLDER" not in os.environ:
    root_folder="/home/guillefix/code/inria/captionRLenv/"
else:
    root_folder = os.environ["ROOT_FOLDER"]
if "DATA_FOLDER" not in os.environ:
    data_folder="/home/guillefix/code/inria/UR5/"
else:
    data_folder = os.environ["DATA_FOLDER"]
if "PROCESSED_DATA_FOLDER" not in os.environ:
    processed_data_folder="/home/guillefix/code/inria/UR5_processed/"
else:
    processed_data_folder=os.environ["PROCESSED_DATA_FOLDER"]
if "ROOT_DIR_MODEL" not in os.environ:
    root_dir_model = "/home/guillefix/code/multimodal-transflower"
else:
    root_dir_model = os.environ["ROOT_DIR_MODEL"]

'''
export ROOT_FOLDER=/mnt/tianwei/captionRLenv/
export DATA_FOLDER=/mnt/tianwei/UR5/
export PROCESSED_DATA_FOLDER=/mnt/tianwei/UR5_processed/
export ROOT_DIR_MODEL=/mnt/multimodal-transflower/
'''

import argparse
parser = argparse.ArgumentParser(description='Evaluate LangGoalRobot environment')
parser.add_argument('--using_model', action='store_true', help='whether to evaluate a model or to evaluate a recorded trajectory')
parser.add_argument('--render', action='store_true', help='whether to render the environment')
parser.add_argument('--session_id', help='the session from which to restore the demo')
parser.add_argument('--rec_id', help='the recording from within the session to retrieve as demo')
parser.add_argument('--experiment_name', default=None, help='experiment name to retrieve the model from (if evaling a model)')
parser.add_argument('--restore_objects', action='store_true', help='whether to restore the objects as they were in the demo')
parser.add_argument('--temp', type=float, default=1.0, help='the temperature parameter for the model (note for normalizing flows, this isnt the real temperature, just a proxy)')
parser.add_argument('--dynamic_temp', action='store_true', help='whether to use the dynamic temperature trick to encourage exploration')
parser.add_argument('--dynamic_temp_delta', type=float, default=0.99, help='the decay/smoothing parameter in the dynamic temp trick algorithm')
parser.add_argument('--max_number_steps', type=int, default=3000, help='the temperature parameter for the model (note for normalizing flows, this isnt the real temperature, just a proxy)')

args = parser.parse_args()


# LOAD demo
session_id=args.session_id
rec_id=args.rec_id
traj_data = np.load(data_folder+session_id+"/obs_act_etc/"+rec_id+"/data.npz", allow_pickle=True)
goal_str = str(traj_data['goal_str'][0])
print(goal_str)
#tokenize goal_str
import json
d=json.load(open(processed_data_folder+"acts.npy.annotation.class_index.json", "r"))

if args.using_model:

    tokens = []
    words = goal_str.split(" ")
    for i in range(11):
        if i < len(words):
            word = words[i]
            tokens.append(d[word])
        else:
            tokens.append(66)

    import torch

    import sys
    sys.path.append(root_dir_model)
    from inference.generate import load_model_from_logs_path

    default_save_path = "pretrained/"+args.experiment_name
    logs_path = default_save_path
    #load model:
    model, opt = load_model_from_logs_path(logs_path)
    if using_torchscript:
        print("Using torchscript")
        model = torch.jit.load(model_folder+'compiled_jit.pth')

    input_dims = opt["input_dims"].split(",")
    context_size_obs=input_dims[1]
    context_size_act=input_dims[2]

    input_mods = opt["input_mods"].split(",")

    obs_scaler = pickle.load(open(processed_data_folder+input_mods[1]+"_scaler.pkl", "rb"))
    acts_scaler = pickle.load(open(processed_data_folder+input_mods[2]+"_scaler.pkl", "rb"))


    filename = "UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data"
    obs_traj = np.load(data_folder+filename+"."+input_mods[1]+".npy")
    obs_traj_unscaled = obs_scaler.inverse_transform(obs_traj)
    acts_traj = np.load(data_folder+filename+"."+input_mods[2]+".npy")
    acts_traj_unscaled = obs_scaler.inverse_transform(acts_traj)


    # discrete_input = np.load(data_folder+filename+"."+input_mods[0]+".npy")
    if input_dims[0] == 10:
        tokens = np.concatenate([tokens[:1], tokens[2:]])
    elif input_dims[0] == 14:
        obj_types = get_obj_types(obs_traj)
        tokens = np.concatenate([tokens, obj_types])

    obj_index = -1
    if input_mods[1] in ["obs_cont_single_nocol_noarm_trim_scaled", "obs_cont_single_nocol_noarm_scaled"]:
        has_conc_obj, color, object_type = has_concrete_object_ann(goal_str)
        print(color, object_type)
        # assert has_conc_obj
        exact_one_object, obj_index = check_if_exact_one_object_obs(obs_cont, disc_cond, color, object_type)
        # assert exact_one_object

    if zero_seed:
        prev_obs = obs_scaler.inverse_transform(np.zeros((context_size_obs,18)))
        prev_acts = acts_scaler.inverse_transform(np.zeros((context_size_act,8)))
    else:
        prev_obs = obs_traj_unscaled[:context_size_obs]
        prev_acts = obs_traj_unscaled[:context_size_acts]

    prev_acts2 = traj_data['acts'][:20]

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

    if using_torchscript:
        inputs = make_inputs(tokens, prev_obs, prev_acts)
        out = model(inputs)
        out = model(inputs)
        print(out.shape)
        print("Prepared torchscript model")

##### START ENV

import os
from src.envs.envList import *
import numpy as np
import pybullet as p

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

object_types = pickle.load(open(root_folder+"object_types.pkl","rb"))
env.env_params['types'] = object_types

#%%
from src.envs.descriptions import generate_all_descriptions
from src.envs.env_params import get_env_params
if args.render:
    env.render(mode='human')
env.reset(o=traj_data["obs"][0], info_reset=None, description=goal_str, joint_poses=traj_data["joint_poses"][0], objects=traj_data['obj_stuff'][0], restore_objs=args.restore_objects)

from src.envs.reward_function import get_reward_from_state

# print([o for o in env.instance.objects])
if joint_control:
    add_joint_controls(env)
else:
    controls = add_xyz_rpy_controls(env)

args.dynamic_temp_delta=0.99
achieved_goal_end=False
achieved_goal_anytime=False
for i in range(args.max_number_steps):

    if joint_control:
        poses  = []
        for i in range(len(env.instance.restJointPositions)):
            poses.append(env.p.readUserDebugParameter(i))
        # Uses a hard reset of the arm joints so that we can quickly debug without worrying about forces
        env.instance.reset_arm_joints(env.instance.arm, poses)

    else:
        if args.using_model:
            if using_torchscript:
                acts = model(inputs)[0][0][0].cpu()
            else:
                variance = np.max(np.abs(prev_acts2[0]-prev_acts2[-1]))
                if args.dynamic_temp:
                    temp = np.max([temp*temp_delta +(1-temp_delta)*10*np.tanh(0.01/variance), 0.5])
                else:
                    temp = args.temp
                acts = model(inputs, temp=temp)[0][0][0].cpu()
            acts = acts_scaler.inverse_transform(acts)
            acts = acts[0]
        else:
            if i>len(traj_data['acts'])-1:
                break
            acts = traj_data['acts'][i]

        action = [acts[0],acts[1],acts[2]] + list(p.getEulerFromQuaternion(acts[3:7])) + [acts[7]]
        # print(action)
        state = env.instance.calc_actor_state()
        obs, r, done, info = env.step(np.array(action))

        # obs[8:11] = [-0.3,0,max_size/2]
        # obs[8:11] = [-0.6,0,0.08]
        # obs[8:11] = [0.6,0.6,0.24]
        # obs[8:11] = [-0.17,0.14,-0.14]
        # env.instance.reset_objects(obs)

        if i == 0:
            initial_state = obs
        else:
            current_state = obs
            success = get_reward_from_state(initial_state, current_state, traj_data['obj_stuff'], goal_str, env.instance.env_params)
            print(goal_str+": ",success)
            achieved_goal_end = success
            if success:
                achieved_goal_anytime = True
            if args.using_model:
                if success:
                    break

        if args.using_model:
            new_obs = obs
            if input_mods[1][:8] == "obs_cont":
                new_obs = get_obs_cont(obs)
                if "single" in input_mods[1]:
                    nocol = "nocol" in input_mods[1]
                    noarm = "noarm" in input_mods[1]
                    new_obs = get_new_obs_obs(new_obs[None], obj_index, nocol=nocol, noarm=noarm)[0]
            prev_obs = np.concatenate([prev_obs[1:],new_obs[None]])
            # print(new_obs)
            prev_acts = np.concatenate([prev_acts[1:],acts[None]])
            prev_acts2 = np.concatenate([prev_acts2[1:],acts[None]])
            inputs = make_inputs(tokens, prev_obs, prev_acts)

if not Path(root_folder+"results").is_dir():
    os.mkdir(root_folder+"results")
if args.using_model:
    filename = root_folder+"results/eval_"+args.experiment_name+"_"+args.session_id+"_"+args.rec_id+"_"+"_".join(goal_str.split(" "))+"_"+str(args.restore_objects)+".txt"
    if os.path.exists(filename):
        with open(filename, "a") as f:
            f.write(str(achieved_goal_end)+","+str(i)+"\n")
    else:
        with open(filename, "w") as f:
            f.write("achieved_goal_end,num_steps"+"\n")
            f.write(str(achieved_goal_end)+","+str(i)+"\n")
else:
    if not Path(root_folder+"results/eval_demos").is_dir():
        os.mkdir(root_folder+"results/eval_demos")
    filename = root_folder+"results/eval_demos/"+args.session_id+"_"+args.rec_id+"_"+"_".join(goal_str.split(" "))+"_"+str(args.restore_objects)+".txt"
    if achieved_goal_anytime:
        with open(root_folder+"results/eval_demos/achieved_goal_anytime.txt", "a") as f:
            f.write("UR5_"+args.session_id+"_obs_act_etc_"+args.rec_id+"_data"+","+goal_str+"\n")
    if achieved_goal_end:
        with open(root_folder+"results/eval_demos/achieved_goal_end.txt", "a") as f:
            f.write("UR5_"+args.session_id+"_obs_act_etc_"+args.rec_id+"_data"+","+goal_str+"\n")
    if not achieved_goal_anytime:
        with open(root_folder+"results/eval_demos/not_achieved_goal_anytime.txt", "a") as f:
            f.write("UR5_"+args.session_id+"_obs_act_etc_"+args.rec_id+"_data"+","+goal_str+"\n")
    if not achieved_goal_end:
        with open(root_folder+"results/eval_demos/not_achieved_goal_end.txt", "a") as f:
            f.write("UR5_"+args.session_id+"_obs_act_etc_"+args.rec_id+"_data"+","+goal_str+"\n")
    if os.path.exists(filename):
        with open(filename, "a") as f:
            f.write(str(achieved_goal_anytime)+","+str(achieved_goal_end)+","+str(i)+"\n")
    else:
        with open(filename, "a") as f:
            f.write("achieved_goal_anytime,achieved_goal_end,num_steps"+"\n")
            f.write(str(achieved_goal_anytime)+","+str(achieved_goal_end)+","+str(i)+"\n")
