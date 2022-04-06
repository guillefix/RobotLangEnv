import numpy as np
import time
import torch.nn.functional as F
import pickle
import json
from pathlib import Path
import os
from src.envs.envList import *
import numpy as np
import pybullet as p

from src.envs.env_params import get_env_params
from src.envs.color_generation import infer_color
from extra_utils.data_utils import get_obs_cont, fix_quaternions, one_hot
from create_simple_dataset import has_concrete_object_ann, check_if_exact_one_object_from_obs, get_new_obs_from_obs
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
parser.add_argument('--save_eval_results', action='store_true', help='whether to save evaluation results')
parser.add_argument('--save_sampled_traj', action='store_true', help='whether to save the sampled trajectory (really only makes sense if using_model)')
parser.add_argument('--render', action='store_true', help='whether to render the environment')
parser.add_argument('--goal_str', help='specify goal string (if not specified, we use the one from the demo)')
parser.add_argument('--zero_seed', action='store_true', help='whether to seed the obs and acts with zeros or with the beginning of the demo')
parser.add_argument('--random_seed', action='store_true', help='whether to seed the obs and acts with a standard normal distribution')
parser.add_argument('--using_torchscript', action='store_true', help='whether to use torchscript compiled model or not')
parser.add_argument('--session_id', help='the session from which to restore the demo')
parser.add_argument('--rec_id', help='the recording from within the session to retrieve as demo')
parser.add_argument('--pretrained_name', default=None, help='experiment name to retrieve the model from (if evaling a model)')
parser.add_argument('--experiment_name', default="default", help='experiment name to save results')
parser.add_argument('--restore_objects', action='store_true', help='whether to restore the objects as they were in the demo')
parser.add_argument('--temp', type=float, default=1.0, help='the temperature parameter for the model (note for normalizing flows, this isnt the real temperature, just a proxy)')
parser.add_argument('--dynamic_temp', action='store_true', help='whether to use the dynamic temperature trick to encourage exploration')
parser.add_argument('--dynamic_temp_delta', type=float, default=0.99, help='the decay/smoothing parameter in the dynamic temp trick algorithm')
parser.add_argument('--max_number_steps', type=int, default=3000, help='the temperature parameter for the model (note for normalizing flows, this isnt the real temperature, just a proxy)')


def run(using_model=False, render=False, goal_str=None, session_id=None, rec_id=None, pretrained_name=None, experiment_name=None, restore_objects=False, temp=1.0, dynamic_temp=False, dynamic_temp_delta=0.99, max_number_steps=3000, zero_seed=False, random_seed=False, using_torchscript=False, save_eval_results=False, save_sampled_traj=False):
    # LOAD demo
    traj_data = np.load(data_folder+session_id+"/obs_act_etc/"+rec_id+"/data.npz", allow_pickle=True)
    if goal_str is None:
        goal_str = str(traj_data['goal_str'][0])
    print(goal_str)


    if using_model:


        import torch

        import sys
        sys.path.append(root_dir_model)
        from inference.generate import load_model_from_logs_path

        default_save_path = pretrained_folder+experiment_name
        logs_path = default_save_path
        #load model:
        model, opt = load_model_from_logs_path(logs_path)
        if using_torchscript:
            print("Using torchscript")
            model = torch.jit.load(model_folder+'compiled_jit.pth')

        input_dims = [int(x) for x in opt.dins.split(",")]
        input_lengths = [int(x) for x in opt.input_lengths.split(",")]
        context_size_obs=input_lengths[1]
        context_size_acts=input_lengths[2]

        input_mods = opt.input_modalities.split(",")

        obs_scaler = pickle.load(open(processed_data_folder+input_mods[1]+"_scaler.pkl", "rb"))
        acts_scaler = pickle.load(open(processed_data_folder+input_mods[2]+"_scaler.pkl", "rb"))


        filename = "UR5_"+session_id+"_obs_act_etc_"+rec_id+"_data"
        obs_traj = np.load(processed_data_folder+filename+"."+input_mods[1]+".npy")
        obs_traj_unscaled = obs_scaler.inverse_transform(obs_traj)
        acts_traj = np.load(processed_data_folder+filename+"."+input_mods[2]+".npy")
        acts_traj_unscaled = acts_scaler.inverse_transform(acts_traj)



        if zero_seed:
            prev_obs = obs_scaler.inverse_transform(np.zeros((context_size_obs,18)))
            prev_acts = acts_scaler.inverse_transform(np.zeros((context_size_acts,8)))
        elif random_seed:
            prev_obs = obs_scaler.inverse_transform(np.random.randn(context_size_obs,18))
            prev_acts = acts_scaler.inverse_transform(np.random.randn(context_size_acts,8))
        else:
            prev_obs = obs_traj_unscaled[:context_size_obs]
            prev_acts = acts_traj_unscaled[:context_size_acts]

        prev_acts2 = traj_data['acts'][:20]

        def scale_inputs(prev_obs, prev_acts, noarm=True):
            if not noarm:
                prev_obs[3:7] = fix_quaternions(prev_obs[3:7])
            prev_obs = obs_scaler.transform(prev_obs)
            prev_acts[3:7] = fix_quaternions(prev_acts[3:7])
            prev_acts = acts_scaler.transform(prev_acts)
            return prev_obs, prev_acts

        def make_inputs(tokens, prev_obs, prev_acts):
            # return [torch.from_numpy(tokens.copy()).unsqueeze(1).unsqueeze(1).cuda(), torch.from_numpy(prev_obs.copy()).unsqueeze(1).float().cuda(), torch.from_numpy(prev_acts.copy()).unsqueeze(1).float().cuda()]
            tokens = torch.from_numpy(tokens)
            # tokens = F.one_hot(tokens,num_classes=67)
            tokens = tokens.unsqueeze(1).unsqueeze(1).long().cuda()
            # tokens = tokens.unsqueeze(1).cuda()
            return [tokens, torch.from_numpy(prev_obs).unsqueeze(1).float().cuda(), torch.from_numpy(prev_acts).unsqueeze(1).float().cuda()]
            # return [torch.from_numpy(tokens).unsqueeze(1).unsqueeze(1).cpu(), torch.from_numpy(prev_obs).unsqueeze(1).float().cpu(), torch.from_numpy(prev_acts).unsqueeze(1).float().cpu()]

        if using_torchscript:
            out = model(inputs)
            out = model(inputs)
            print(out.shape)
            print("Prepared torchscript model")

    ##### START ENV


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
    if render:
        env.render(mode='human')

    objects = traj_data['obj_stuff'][0] if restore_objects else None
    env.reset(o=traj_data["obs"][0], info_reset=None, description=goal_str, joint_poses=traj_data["joint_poses"][0], objects=objects, restore_objs=restore_objects)

    #prepare first inputs
    obj_stuff = env.instance.get_stuff_to_save()
    tokens = get_tokens(goal_str, input_lengths, obj_stuff)
    prev_obs, prev_acts = scale_inputs(prev_obs, prev_acts, "noarm" in input_mods[1])
    inputs = make_inputs(tokens, prev_obs, prev_acts)
    print("obj_stuff")
    print(obj_stuff)

    if using_model:
        obj_index = -1
        if input_mods[1] in ["obs_cont_single_nocol_noarm_trim_scaled", "obs_cont_single_nocol_noarm_scaled"]:
            has_conc_obj, color, object_type = has_concrete_object_ann(goal_str)
            print(color, object_type)
            # assert has_conc_obj
            # exact_one_object, obj_index = check_if_exact_one_object_from_obs(obs_cont, disc_cond, color, object_type)
            objects = env.instance.objects_added
            matches = 0
            for i, obj in enumerate(objects):
                if obj['type'] == object_type and obj['color'] == color:
                    matches += 1
                    obj_index = i
        print("obj_index", obj_index)
            # assert exact_one_object


    # print([o for o in env.instance.objects])
    if joint_control:
        add_joint_controls(env)
    else:
        controls = add_xyz_rpy_controls(env)

    # dynamic_temp_delta=0.99
    achieved_goal_end=False
    achieved_goal_anytime=False
    if save_sampled_traj:
        obss = []
        actss = []
        joints = []
        acts_rpy = []
        acts_rpy_rel = []
        velocities = []
        targetJoints = []
        gripper_proprioception = []
        if using_model:
            scaled_obss = None
            scaled_actss = None
    for i in range(max_number_steps):

        if joint_control:
            poses  = []
            for i in range(len(env.instance.restJointPositions)):
                poses.append(env.p.readUserDebugParameter(i))
            # Uses a hard reset of the arm joints so that we can quickly debug without worrying about forces
            env.instance.reset_arm_joints(env.instance.arm, poses)

        else:
            if using_model:
                if using_torchscript:
                    acts = model(inputs)[0][0][0].cpu()
                else:
                    variance = np.max(np.abs(prev_acts2[0]-prev_acts2[-1]))
                    if dynamic_temp:
                        temp = np.max([temp*dynamic_temp_delta +(1-dynamic_temp_delta)*10*np.tanh(0.01/variance), 0.5])
                    else:
                        temp = temp
                    scaled_acts = model(inputs, temp=temp)[0][0][0].cpu()
                acts = acts_scaler.inverse_transform(scaled_acts)
                acts = acts[0]
            else:
                if i>len(traj_data['acts'])-1:
                    break
                acts = traj_data['acts'][i]

            act_pos = [acts[0],acts[1],acts[2]]
            act_gripper = [acts[7]]
            acts_euler = list(p.getEulerFromQuaternion(acts[3:7]))
            action = act_pos + acts_euler + act_gripper
            # print(action)
            if save_sampled_traj:
                # state = env.instance.calc_actor_state()
                state = env.instance.calc_state()
                # print(state)
                rel_xyz = np.array(act_pos)-np.array(state['observation'][0:3])
                rel_rpy = np.array(acts_euler) - np.array(p.getEulerFromQuaternion(state['observation'][3:7]))
                action_rpy_rel = np.array(list(rel_xyz)+list(rel_rpy)+act_gripper)
                actss.append(acts)
                obss.append(state['observation'])
                joints.append(state['joints'])
                acts_rpy.append(action)
                acts_rpy_rel.append(action_rpy_rel)
                velocities.append(state['velocity'])
                gripper_proprioception.append(state['gripper_proprioception'])
            obs, r, done, info = env.step(np.array(action))
            if save_sampled_traj:
                targetJoints.append(info["target_poses"])
            # if using_model:
            #     print(obs[8+35*obj_index:11+35*obj_index])
            #     print(obs[113])

            # obs[8:11] = [-0.3,0,max_size/2]
            # obs[8:11] = [-0.6,0,0.08]
            # obs[8:11] = [0.6,0.6,0.24]
            # obs[8:11] = [-0.3,0.4,0.04]
            # env.instance.reset_objects(obs)

            if i == 0:
                initial_state = obs
            else:
                current_state = obs
                success = get_reward_from_state(initial_state, current_state, obj_stuff, goal_str, env.instance.env_params)
                print(goal_str+": ",success)
                achieved_goal_end = success
                if success:
                    achieved_goal_anytime = True
                if using_model:
                    if success:
                        break

            if using_model:
                new_obs = obs
                if input_mods[1][:8] == "obs_cont":
                    new_obs = get_obs_cont(obs)
                    if "single" in input_mods[1]:
                        nocol = "nocol" in input_mods[1]
                        noarm = "noarm" in input_mods[1]
                        new_obs = get_new_obs_from_obs(new_obs[None], obj_index, nocol=nocol, noarm=noarm)[0]

                prev_acts2 = np.concatenate([prev_acts2[1:],acts[None]])
                new_obs, acts = scale_inputs(new_obs[None], acts[None], "noarm" in input_mods[1])
                prev_obs = np.concatenate([prev_obs[1:],new_obs])
                # print(new_obs)
                prev_acts = np.concatenate([prev_acts[1:],acts])
                inputs = make_inputs(tokens, prev_obs, prev_acts)

            if save_sampled_traj and using_model:
                if scaled_obss is None:
                    scaled_obss = new_obs
                else:
                    scaled_obss = np.concatenate([scaled_obss, new_obs])
                if scaled_actss is None:
                    scaled_actss = acts
                else:
                    scaled_actss = np.concatenate([scaled_actss, acts])


    if save_sampled_traj:
        train_descriptions, test_descriptions = sample_descriptions_from_state(initial_state, current_state, obj_stuff, env.instance.env_params)
        descriptions = train_descriptions + test_descriptions
        print(descriptions)
        if len(descriptions)>0:
            description = descriptions[0]
            new_session_id = experiment_name
            new_rec_id = str(uuid.uuid4())
            if not Path(root_folder+"generated_data").is_dir():
                os.mkdir(root_folder+"generated_data")
            if not Path(root_folder+"generated_data/"+new_session_id).is_dir():
                os.mkdir(root_folder+"generated_data/"+new_session_id)
            if not Path(root_folder+"generated_data/"+new_session_id+"/"+new_rec_id).is_dir():
                os.mkdir(root_folder+"generated_data/"+new_session_id+"/"+new_rec_id)
            npz_path = root_folder+"generated_data/"+new_session_id+"/"+new_rec_id
            save_traj(npz_path, acts, obs, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception, [description], obj_stuff)
            args_file = root_folder+"generated_data/"+new_session_id+"/"+new_rec_id+"/args.json"
            json_string = json.dumps(args)
            with open(args_file, "w") as f:
                f.write(json_string)
            if using_model:
                new_tokens = get_tokens(goal_str, input_lengths, obj_stuff)
                if not Path(root_folder+"generated_data_processed").is_dir():
                    os.mkdir(root_folder+"generated_data_processed")
                np.save(root_folder+"generated_data_processed/"+"UR5_{}_obs_act_etc_{}_data".format(new_session_id, new_rec_id)+"."+input_mods[0], new_tokens)
                np.save(root_folder+"generated_data_processed/"+"UR5_{}_obs_act_etc_{}_data".format(new_session_id, new_rec_id)+"."+input_mods[1], obss)
                np.save(root_folder+"generated_data_processed/"+"UR5_{}_obs_act_etc_{}_data".format(new_session_id, new_rec_id)+"."+input_mods[2], actss)
    if save_eval_results:
        if not Path(root_folder+"results").is_dir():
            os.mkdir(root_folder+"results")
        if not Path(root_folder+"results/"+experiment_name).is_dir():
            os.mkdir(root_folder+"results/"+experiment_name)
        if using_model:
            filename = root_folder+"results/"+experiment_name+"/eval_"+experiment_name+"_"+session_id+"_"+rec_id+"_"+"_".join(goal_str.split(" "))+"_"+str(restore_objects)+".txt"
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
