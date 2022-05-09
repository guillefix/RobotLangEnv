import numpy as np
import time
import torch.nn.functional as F
import pickle
import json
from pathlib import Path
import os
from src.envs.envList import *
import numpy as np
import torch
import pybullet as p

from src.envs.env_params import get_env_params
from src.envs.color_generation import infer_color
from extra_utils.data_utils import get_tokens
from create_simple_dataset import has_concrete_object_ann, check_if_exact_one_object_from_obs
from src.envs.utils import save_traj
import uuid
from src.envs.reward_function import get_reward_from_state, sample_descriptions_from_state

from constants import *

'''
export ROOT_FOLDER=/mnt/tianwei/captionRLenv/
export PRETRAINED_FOLDER=/mnt/tianwei/pretrained/
export DATA_FOLDER=/mnt/tianwei/UR5/
export PROCESSED_DATA_FOLDER=/mnt/tianwei/UR5_processed/
export ROOT_DIR_MODEL=/mnt/multimodal-transflower/
'''

import argparse
parser = argparse.ArgumentParser(description='Evaluate LangGoalRobot environment')
parser.add_argument('--using_model', action='store_true', help='whether to evaluate a model or to evaluate a recorded trajectory')
parser.add_argument('--computing_loss', action='store_true', help='whether to compute the loss of a recorded trajectory')
parser.add_argument('--computing_relabelled_logPs', action='store_true', help='whether to compute the logP of the trajectory chunk before a state where some goal(s) have been completed')
parser.add_argument('--save_eval_results', action='store_true', help='whether to save evaluation results')
parser.add_argument('--save_relabelled_trajs', action='store_true', help='whether to save the relabelled subtrajectories')
parser.add_argument('--render', action='store_true', help='whether to render the environment')
parser.add_argument('--goal_str', help='specify goal string (if not specified, we use the one from the demo)')
parser.add_argument('--zero_seed', action='store_true', help='whether to seed the obs and acts with zeros or with the beginning of the demo')
parser.add_argument('--random_seed', action='store_true', help='whether to seed the obs and acts with a standard normal distribution')
parser.add_argument('--using_torchscript', action='store_true', help='whether to use torchscript compiled model or not')
parser.add_argument('--session_id', help='the session from which to restore the demo')
parser.add_argument('--rec_id', help='the recording from within the session to retrieve as demo')
parser.add_argument('--pretrained_name', default=None, help='experiment name to retrieve the model from (if evaling a model)')
parser.add_argument('--experiment_name', default="default", help='experiment name to save results')
parser.add_argument('--varying_args', default="session_id,rec_id", help='comma-separated list of arguments that vary in the experiment')
parser.add_argument('--restore_objects', action='store_true', help='whether to restore the objects as they were in the demo')
parser.add_argument('--temp', type=float, default=1.0, help='the temperature parameter for the model (note for normalizing flows, this isnt the real temperature, just a proxy)')
parser.add_argument('--dynamic_temp', action='store_true', help='whether to use the dynamic temperature trick to encourage exploration')
parser.add_argument('--dynamic_temp_delta', type=float, default=0.99, help='the decay/smoothing parameter in the dynamic temp trick algorithm')
parser.add_argument('--save_chunk_size', type=int, default=120, help='the number of frames previous to achievement of a goal to consider part of the episode saved for that goal')
parser.add_argument('--max_number_steps', type=int, default=3000, help='the temperature parameter for the model (note for normalizing flows, this isnt the real temperature, just a proxy)')
parser.add_argument('--times_to_go_start', type=int, default=200, help='the initial times to go when set manually')

if torch.cuda.is_available():
    print("CUDA available")
    # device = 'gpu:'+str(torch.cuda.current_device())
    device = 'cuda'
else:
    print("CUDA not available")
    device = 'cpu'


def run(using_model=False, computing_loss=False, computing_relabelled_logPs=False, render=False, goal_str=None, session_id=None, rec_id=None, pretrained_name=None, experiment_name=None, restore_objects=False, temp=1.0, dynamic_temp=False, dynamic_temp_delta=0.99, max_number_steps=3000, zero_seed=False, random_seed=False, using_torchscript=False, save_eval_results=False, save_relabelled_trajs=False, varying_args="session_id,rec_id", save_chunk_size=120, times_to_go_start=200):
    varying_args = varying_args.split(",")
    args = locals()
    # LOAD demo data
    if os.path.exists(data_folder+session_id+"/obs_act_etc/"+rec_id+"/data.npz"):
        traj_data = np.load(data_folder+session_id+"/obs_act_etc/"+rec_id+"/data.npz", allow_pickle=True)
    elif os.path.exists(data_folder+session_id+"/"+rec_id+"/data.npz"):
        traj_data = np.load(data_folder+session_id+"/"+rec_id+"/data.npz", allow_pickle=True)
        print(traj_data)
    if goal_str is None:
        goal_str = str(traj_data['goal_str'][0])
    print(goal_str)

    if using_model or computing_loss or computing_relabelled_logPs:
        import torch

        import sys
        sys.path.append(root_dir_model)
        from inference.generate import load_model_from_logs_path

        default_save_path = pretrained_folder+pretrained_name
        logs_path = default_save_path
        #load model:
        model, opt = load_model_from_logs_path(logs_path)
        if using_torchscript:
            print("Using torchscript")
            model = torch.jit.load(model_folder+'compiled_jit.pth')
        else:
            model = model.to(device)

        input_dims = [int(x) for x in str(opt.dins).split(",")]
        output_dims = [int(x) for x in str(opt.douts).split(",")]
        input_lengths = [int(x) for x in str(opt.input_lengths).split(",")]

        input_mods = opt.input_modalities.split(",")
        output_mods = opt.output_modalities.split(",")

        obs_mod = None
        obs_mod_idx = None
        acts_mod = None
        acts_mod_idx = None
        ann_mod = None
        ann_mod_idx = None
        ttg_mod = None
        ttg_mod_idx = None
        for i,mod in enumerate(input_mods):
            if "obs" in mod:
                obs_mod = mod
                obs_mod_idx = i
            elif "acts" in mod:
                acts_mod = mod
                acts_mod_idx = i
            elif "annotation" in mod:
                ann_mod = mod
                ann_mod_idx = i
            elif "times_to_go" in mod:
                ttg_mod = mod
                ttg_mod_idx = i

        if ttg_mod is None:
            times_to_go = None
        else:
            times_to_go = np.array(range(times_to_go_start+input_lengths[ttg_mod_idx]-1, times_to_go_start-1, -1))
            times_to_go = np.expand_dims(times_to_go, 1)

        context_size_obs=input_lengths[obs_mod_idx]
        context_size_acts=input_lengths[acts_mod_idx]
        if ttg_mod is not None:
            context_size_ttg=input_lengths[ttg_mod_idx]

        #obs_scaler = pickle.load(open(pretrained_folder+pretrained_name+"/"+obs_mod+"_scaler.pkl", "rb"))
        #acts_scaler = pickle.load(open(pretrained_folder+pretrained_name+"/"+acts_mod+"_scaler.pkl", "rb"))
        obs_scaler = pickle.load(open(processed_data_folder+obs_mod+"_scaler.pkl", "rb"))
        acts_scaler = pickle.load(open(processed_data_folder+acts_mod+"_scaler.pkl", "rb"))

        if zero_seed:
            prev_obs = obs_scaler.inverse_transform(np.zeros((context_size_obs,input_dims[obs_mod_idx])))
            prev_acts = acts_scaler.inverse_transform(np.zeros((context_size_acts,output_dims[0])))
        elif random_seed:
            prev_obs = obs_scaler.inverse_transform(np.random.randn(context_size_obs,input_dims[obs_mod_idx]))
            prev_acts = acts_scaler.inverse_transform(np.random.randn(context_size_acts,outpu_dims[0]))
        else:
            # print(processed_data_folder+"generated_data_"+session_id+"_"+rec_id+"_data."+obs_mod+".npy")
            if os.path.exists(processed_data_folder+"UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data."+obs_mod+".npy"):
                filename = "UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data"
            elif os.path.exists(processed_data_folder+"generated_data_"+session_id+"_"+rec_id+"_data."+obs_mod+".npy"):
                filename = "generated_data_"+session_id+"_"+rec_id+"_data"
            obs_traj = np.load(processed_data_folder+filename+"."+obs_mod+".npy")
            obs_traj_unscaled = obs_scaler.inverse_transform(obs_traj)
            acts_traj = np.load(processed_data_folder+filename+"."+acts_mod+".npy")
            acts_traj_unscaled = acts_scaler.inverse_transform(acts_traj)
            prev_obs = obs_traj_unscaled[:context_size_obs]
            prev_acts = acts_traj_unscaled[:context_size_acts]

        if using_model and dynamic_temp:
            prev_acts2 = traj_data['acts'][:20]

        if using_torchscript:
            out = model(inputs)
            out = model(inputs)
            print(out.shape)
            print("Prepared torchscript model")

    ##### START ENV
    joint_control = False # Toggle this flag to control joints or ABS RPY Space
    if using_model:
        env = ExtendedUR5PlayAbsRPY1Obj(obs_scaler = obs_scaler, acts_scaler = acts_scaler, prev_obs = prev_obs, save_relabelled_trajs = save_relabelled_trajs,
                                        prev_acts = prev_acts, times_to_go = times_to_go, desc_max_len = input_lengths[ann_mod_idx], obs_mod = obs_mod, args=args)
    else:
        env = ExtendedUR5PlayAbsRPY1Obj(save_relabelled_trajs = save_relabelled_trajs, args = args)

    #%%
    if render:
        env.render(mode='human')

    objects = traj_data['obj_stuff'][0] if restore_objects else None
    # obs,_,_,_ = env.reset(o=traj_data["obs"][0], info_reset=None, description=goal_str, joint_poses=traj_data["joint_poses"][0], objects=objects, restore_objs=restore_objects)
    obs = env.reset(o=traj_data["obs"][0], info_reset=None, description=goal_str, joint_poses=traj_data["joint_poses"][0], objects=objects, restore_objs=restore_objects)

    #prepare inputs for relabelled logPs (if doing it)
    if computing_relabelled_logPs:
        prev_obs_ext = np.zeros((save_chunk_size+context_size_obs,125))
        prev_acts_ext = np.zeros((save_chunk_size+context_size_acts,8))

    # dynamic_temp_delta=0.99
    achieved_goal_end=False
    achieved_goal_anytime=False
    if using_model:
        scaled_obss = None
        scaled_actss = None

    logPs = []
    for t in range(max_number_steps):
        print(t)

        if using_model:
            if using_torchscript:
                scaled_acts = model(inputs)[0][0][0].cpu()
            else:
                if dynamic_temp:
                    variance = np.max(np.abs(prev_acts2[0]-prev_acts2[-1]))
                    # temp = np.max([temp*dynamic_temp_delta +(1-dynamic_temp_delta)*10*np.tanh(0.01/variance), 0.5])
                    temp = np.max([temp*dynamic_temp_delta +(1-dynamic_temp_delta)*10*np.tanh(0.01/variance), 0.8])
                else:
                    temp = temp
                # start_time = time.time()
                scaled_acts, _, logPs_temp = model(tuple((torch.from_numpy(o).to(model.device) for o in obs)), temp=temp)
                # print("--- Inference time: %s seconds ---" % (time.time() - start_time))
                scaled_acts = scaled_acts[0][0].cpu()
                if computing_loss:
                    logP = logPs_temp[0].cpu().item()
                    logPs.append(logP)
                # print(logP)
            action = scaled_acts
            if dynamic_temp:
                acts = acts_scaler.inverse_transform(scaled_acts)[0]
                if using_model:
                    prev_acts2 = np.concatenate([prev_acts2[1:],acts[None]])
            if save_relabelled_trajs:
                new_obs = env.prev_obs[-1]
                new_acts = env.prev_acts[-1]
                if scaled_obss is None:
                    scaled_obss = new_obs
                else:
                    scaled_obss = np.concatenate([scaled_obss, new_obs])
                if scaled_actss is None:
                   scaled_actss = new_acts
                else:
                    scaled_actss = np.concatenate([scaled_actss, new_acts])
        else:
            if t>len(traj_data['acts'])-1:
                break
            action = traj_data['acts'][t]
            if computing_loss or computing_relabelled_logPs:
                scaled_acts = acts_scaler.transform(action[None])
            if computing_loss:
                # logP = model.training_step({**{"in_"+input_mods[j]: inputs[j].permute(1,0,2) for j in range(len(input_mods))}, "out_"+output_mods[0]: torch.from_numpy(scaled_acts).unsqueeze(0).float().to(device)}, batch_idx=0)
                logP = model.training_step({**{"in_"+input_mods[j]: inputs[j].permute(1,0,2) for j in range(len(input_mods))}, "out_"+output_mods[0]: torch.from_numpy(scaled_acts).unsqueeze(0).unsqueeze(0).float().to(device)}, batch_idx=0, reduce_loss=False)
                logP = logP.cpu().numpy()
                logPs.append(logP)
                # print(logP)

        #run env
        obs, r, success, info = env.step(action)

        new_descriptions = info["new_descriptions"]

        if computing_relabelled_logPs:
            if computing_loss:
                print("mean logP original goal_str: "+str(np.mean(logPs[-save_chunk_size:])))
            raw_obs = info["raw_obs"]
            prev_obs_ext = np.concatenate([prev_obs_ext[1:],raw_obs[None]])
            prev_acts_ext = np.concatenate([prev_acts_ext[1:],scaled_acts[None]])

        if len(new_descriptions) > 0:
            if computing_relabelled_logPs:
                compute_relabelled_logPs(obs_scaler, acts_scaler, t, new_descriptions, env, input_lengths, ann_mod_idx, prev_obs_ext, prev_acts_ext)
            if save_relabelled_trajs and using_model:
                if not Path(root_folder_generated_data+"generated_data_processed").is_dir():
                    os.mkdir(root_folder_generated_data+"generated_data_processed")
                with open(root_folder_generated_data+"generated_data_processed/"+"UR5_{}_obs_act_etc_{}_data".format(new_session_id, new_rec_id)+".annotation.txt", "w") as file:
                    for ii,desc in enumerate(descriptions):
                        new_tokens = get_tokens(desc, max_length=input_lengths[ann_mod_idx], obj_stuff=obj_stuff)[None]
                        if ii == 0:
                            new_tokenss = new_tokens
                        else:
                            new_tokenss = np.concatenate([new_tokenss,new_tokens])
                        file.write(desc)
                #TODO: tidy/generalize this
                times_to_go_save = np.expand_dims(np.array(range(t+1)),1)
                np.save(root_folder_generated_data+"generated_data_processed/"+"UR5_{}_obs_act_etc_{}_data".format(new_session_id, new_rec_id)+"."+ann_mod, new_tokenss)
                np.save(root_folder_generated_data+"generated_data_processed/"+"UR5_{}_obs_act_etc_{}_data".format(new_session_id, new_rec_id)+"."+obs_mod, scaled_obss)
                np.save(root_folder_generated_data+"generated_data_processed/"+"UR5_{}_obs_act_etc_{}_data".format(new_session_id, new_rec_id)+"."+acts_mod, scaled_actss)
                np.save(root_folder_generated_data+"generated_data_processed/"+"UR5_{}_obs_act_etc_{}_data".format(new_session_id, new_rec_id)+"."+"times_to_go", times_to_go_save)

        print(goal_str+": ",success)
        achieved_goal_end = success
        if success:
            print(goal_str+": ",success)
            achieved_goal_anytime = True
        if using_model:
            if success:
                break

    if save_eval_results:
        if not Path(root_folder+"results").is_dir():
            os.mkdir(root_folder+"results")
        if not Path(root_folder+"results/"+experiment_name).is_dir():
            os.mkdir(root_folder+"results/"+experiment_name)
        if using_model:
            filename = root_folder+"results/"+experiment_name+"/eval_"
            for k in varying_args:
                filename += str(args[k])+"_"
            filename += "_".join(goal_str.split(" "))+".txt"
            metadata_filename = root_folder+"results/"+experiment_name+"/metadata.txt"
            if not os.path.exists(metadata_filename):
                args_reduced = args.copy()
                for k in varying_args:
                    del args_reduced[k]
                json_string = json.dumps(args_reduced)
                with open(metadata_filename, "w") as f:
                    f.write(json_string)
            if os.path.exists(filename):
                with open(filename, "a") as f:
                    f.write(str(achieved_goal_end)+","+str(i)+"\n")
            else:
                with open(filename, "w") as f:
                    f.write("achieved_goal_end,num_steps"+"\n")
                    f.write(str(achieved_goal_end)+","+str(i)+"\n")
        else:
            filename = root_folder+"results/"+experiment_name+"/"+session_id+"_"+rec_id+"_"+"_".join(goal_str.split(" "))+"_"+str(restore_objects)+".txt"
            if achieved_goal_anytime:
                with open(root_folder+"results/"+experiment_name+"/achieved_goal_anytime.txt", "a") as f:
                    f.write("UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data"+","+goal_str+"\n")
            if achieved_goal_end:
                with open(root_folder+"results/"+experiment_name+"/achieved_goal_end.txt", "a") as f:
                    f.write("UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data"+","+goal_str+"\n")
            if not achieved_goal_anytime:
                with open(root_folder+"results/"+experiment_name+"/not_achieved_goal_anytime.txt", "a") as f:
                    f.write("UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data"+","+goal_str+"\n")
            if not achieved_goal_end:
                with open(root_folder+"results/"+experiment_name+"/not_achieved_goal_end.txt", "a") as f:
                    f.write("UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data"+","+goal_str+"\n")
            if os.path.exists(filename):
                with open(filename, "a") as f:
                    f.write(str(achieved_goal_anytime)+","+str(achieved_goal_end)+","+str(i)+"\n")
            else:
                with open(filename, "a") as f:
                    f.write("achieved_goal_anytime,achieved_goal_end,num_steps"+"\n")
                    f.write(str(achieved_goal_anytime)+","+str(achieved_goal_end)+","+str(i)+"\n")

if __name__ == "__main__":
    args = vars(parser.parse_args())
    run(**args)
